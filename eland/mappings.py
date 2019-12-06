#  Copyright 2019 Elasticsearch BV
#
#      Licensed under the Apache License, Version 2.0 (the "License");
#      you may not use this file except in compliance with the License.
#      You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#      Unless required by applicable law or agreed to in writing, software
#      distributed under the License is distributed on an "AS IS" BASIS,
#      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#      See the License for the specific language governing permissions and
#      limitations under the License.

import warnings

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import (is_float_dtype, is_bool_dtype, is_integer_dtype, is_datetime_or_timedelta_dtype,
                                       is_string_dtype)


class Mappings:
    """
    General purpose to manage Elasticsearch to/from pandas mappings

    Attributes
    ----------

    _mappings_capabilities: pandas.DataFrame
        A data frame summarising the capabilities of the index mapping

        _source     - is top level field (i.e. not a multi-field sub-field)
        es_dtype    - Elasticsearch field datatype
        pd_dtype    - Pandas datatype
        searchable  - is the field searchable?
        aggregatable- is the field aggregatable?
                                        _source es_dtype    pd_dtype    searchable  aggregatable
        maps-telemetry.min              True    long        int64       True        True
        maps-telemetry.avg              True    float       float64     True        True
        city                            True    text        object      True        False
        user_name                       True    keyword     object      True        True
        origin_location.lat.keyword     False   keyword     object      True        True
        type                            True    keyword     object      True        True
        origin_location.lat             True    text        object      True        False

    """

    def __init__(self,
                 client=None,
                 index_pattern=None,
                 mappings=None):
        """
        Parameters
        ----------
        client: eland.Client
            Elasticsearch client

        index_pattern: str
            Elasticsearch index pattern

        Copy constructor arguments

        mappings: Mappings
            Object to copy
        """

        # here we keep track of the format of any date fields
        self._date_fields_format = {}
        if (client is not None) and (index_pattern is not None):
            get_mapping = client.get_mapping(index=index_pattern)

            # Get all fields (including all nested) and then all field_caps
            all_fields, self._date_fields_format = Mappings._extract_fields_from_mapping(get_mapping)
            all_fields_caps = client.field_caps(index=index_pattern, fields='*')

            # Get top level (not sub-field multifield) mappings
            source_fields, _ = Mappings._extract_fields_from_mapping(get_mapping, source_only=True)

            # Populate capability matrix of fields
            # field_name, es_dtype, pd_dtype, is_searchable, is_aggregtable, is_source
            self._mappings_capabilities = Mappings._create_capability_matrix(all_fields, source_fields, all_fields_caps)
        else:
            # straight copy
            self._mappings_capabilities = mappings._mappings_capabilities.copy()

        # Cache source field types for efficient lookup
        # (this massively improves performance of DataFrame.flatten)
        self._source_field_pd_dtypes = {}

        for field_name in self._mappings_capabilities[self._mappings_capabilities._source].index:
            pd_dtype = self._mappings_capabilities.loc[field_name]['pd_dtype']
            self._source_field_pd_dtypes[field_name] = pd_dtype

    @staticmethod
    def _extract_fields_from_mapping(mappings, source_only=False, date_format=None):
        """
        Extract all field names and types from a mapping.
        ```
        {
          "my_index": {
            "mappings": {
              "properties": {
                "city": {
                  "type": "text",
                  "fields": {
                    "keyword": {
                      "type": "keyword"
                    }
                  }
                }
              }
            }
          }
        }
        ```
        if source_only == False:
            return {'city': 'text', 'city.keyword': 'keyword'}
        else:
            return {'city': 'text'}

        Note: first field name type wins. E.g.

        ```
        PUT my_index1 {"mappings":{"properties":{"city":{"type":"text"}}}}
        PUT my_index2 {"mappings":{"properties":{"city":{"type":"long"}}}}

        Returns {'city': 'text'}
        ```

        Parameters
        ----------
        mappings: dict
            Return from get_mapping

        Returns
        -------
        fields, dates_format: tuple(dict, dict)
            where:
                fields: Dict of field names and types
                dates_format: Dict of date field names and format

        """
        fields = {}
        dates_format = {}

        # Recurse until we get a 'type: xxx'
        def flatten(x, name=''):
            if type(x) is dict:
                for a in x:
                    if a == 'type' and type(x[a]) is str:  # 'type' can be a name of a field
                        field_name = name[:-1]
                        field_type = x[a]
                        # if field_type is 'date' keep track of the format info when available
                        if field_type == "date" and "format" in x:
                            dates_format[field_name] = x["format"]
                        # If there is a conflicting type, warn - first values added wins
                        if field_name in fields and fields[field_name] != field_type:
                            warnings.warn("Field {} has conflicting types {} != {}".
                                          format(field_name, fields[field_name], field_type),
                                          UserWarning)
                        else:
                            fields[field_name] = field_type
                    elif a == 'properties' or (not source_only and a == 'fields'):
                        flatten(x[a], name)
                    elif not (source_only and a == 'fields'):  # ignore multi-field fields for source_only
                        flatten(x[a], name + a + '.')

        for index in mappings:
            if 'properties' in mappings[index]['mappings']:
                properties = mappings[index]['mappings']['properties']

                flatten(properties)

        return fields, dates_format

    @staticmethod
    def _create_capability_matrix(all_fields, source_fields, all_fields_caps):
        """
        {
          "fields": {
            "rating": {
              "long": {
                "searchable": true,
                "aggregatable": false,
                "indices": ["index1", "index2"],
                "non_aggregatable_indices": ["index1"]
              },
              "keyword": {
                "searchable": false,
                "aggregatable": true,
                "indices": ["index3", "index4"],
                "non_searchable_indices": ["index4"]
              }
            },
            "title": {
              "text": {
                "searchable": true,
                "aggregatable": false

              }
            }
          }
        }
        """
        all_fields_caps_fields = all_fields_caps['fields']

        field_names = ['_source', 'es_dtype', 'pd_dtype', 'searchable', 'aggregatable']
        capability_matrix = {}

        for field, field_caps in all_fields_caps_fields.items():
            if field in all_fields:
                # v = {'long': {'type': 'long', 'searchable': True, 'aggregatable': True}}
                for kk, vv in field_caps.items():
                    _source = (field in source_fields)
                    es_dtype = vv['type']
                    pd_dtype = Mappings._es_dtype_to_pd_dtype(vv['type'])
                    searchable = vv['searchable']
                    aggregatable = vv['aggregatable']

                    caps = [_source, es_dtype, pd_dtype, searchable, aggregatable]

                    capability_matrix[field] = caps

                    if 'non_aggregatable_indices' in vv:
                        warnings.warn("Field {} has conflicting aggregatable fields across indexes {}",
                                      format(field, vv['non_aggregatable_indices']),
                                      UserWarning)
                    if 'non_searchable_indices' in vv:
                        warnings.warn("Field {} has conflicting searchable fields across indexes {}",
                                      format(field, vv['non_searchable_indices']),
                                      UserWarning)

        capability_matrix_df = pd.DataFrame.from_dict(capability_matrix, orient='index', columns=field_names)

        return capability_matrix_df.sort_index()

    @staticmethod
    def _es_dtype_to_pd_dtype(es_dtype):
        """
        Mapping Elasticsearch types to pandas dtypes
        --------------------------------------------

        Elasticsearch field datatype              | Pandas dtype
        --
        text                                      | object
        keyword                                   | object
        long, integer, short, byte, binary        | int64
        double, float, half_float, scaled_float   | float64
        date, date_nanos                          | datetime64
        boolean                                   | bool
        TODO - add additional mapping types
        """
        es_dtype_to_pd_dtype = {
            'text': 'object',
            'keyword': 'object',

            'long': 'int64',
            'integer': 'int64',
            'short': 'int64',
            'byte': 'int64',
            'binary': 'int64',

            'double': 'float64',
            'float': 'float64',
            'half_float': 'float64',
            'scaled_float': 'float64',

            'date': 'datetime64[ns]',
            'date_nanos': 'datetime64[ns]',

            'boolean': 'bool'
        }

        if es_dtype in es_dtype_to_pd_dtype:
            return es_dtype_to_pd_dtype[es_dtype]

        # Return 'object' for all unsupported TODO - investigate how different types could be supported
        return 'object'

    @staticmethod
    def _pd_dtype_to_es_dtype(pd_dtype):
        """
        Mapping pandas dtypes to Elasticsearch dtype
        --------------------------------------------

        ```
        Pandas dtype	Python type	NumPy type	Usage
        object	str	string_, unicode_	Text
        int64	int	int_, int8, int16, int32, int64, uint8, uint16, uint32, uint64	Integer numbers
        float64	float	float_, float16, float32, float64	Floating point numbers
        bool	bool	bool_	True/False values
        datetime64	NA	datetime64[ns]	Date and time values
        timedelta[ns]	NA	NA	Differences between two datetimes
        category	NA	NA	Finite list of text values
        ```
        """
        es_dtype = None

        # Map all to 64-bit - TODO map to specifics: int32 -> int etc.
        if is_float_dtype(pd_dtype):
            es_dtype = 'double'
        elif is_integer_dtype(pd_dtype):
            es_dtype = 'long'
        elif is_bool_dtype(pd_dtype):
            es_dtype = 'boolean'
        elif is_string_dtype(pd_dtype):
            es_dtype = 'keyword'
        elif is_datetime_or_timedelta_dtype(pd_dtype):
            es_dtype = 'date'
        else:
            warnings.warn('No mapping for pd_dtype: [{0}], using default mapping'.format(pd_dtype))

        return es_dtype

    @staticmethod
    def _generate_es_mappings(dataframe, geo_points=None):
        """Given a pandas dataframe, generate the associated Elasticsearch mapping

        Parameters
        ----------
            dataframe : pandas.DataFrame
                pandas.DataFrame to create schema from

        Returns
        -------
            mapping : str
        """

        """
        "mappings" : {
          "properties" : {
            "AvgTicketPrice" : {
              "type" : "float"
            },
            "Cancelled" : {
              "type" : "boolean"
            },
            "Carrier" : {
              "type" : "keyword"
            },
            "Dest" : {
              "type" : "keyword"
            }
          }
        }
        """

        mappings = {'properties': {}}
        for field_name_name, dtype in dataframe.dtypes.iteritems():
            if geo_points is not None and field_name_name in geo_points:
                es_dtype = 'geo_point'
            else:
                es_dtype = Mappings._pd_dtype_to_es_dtype(dtype)

            mappings['properties'][field_name_name] = {}
            mappings['properties'][field_name_name]['type'] = es_dtype

        return {"mappings": mappings}

    def all_fields(self):
        """
        Returns
        -------
        all_fields: list
            All typed fields in the index mapping
        """
        return self._mappings_capabilities.index.tolist()

    def field_capabilities(self, field_name):
        """
        Parameters
        ----------
        field_name: str

        Returns
        -------
        mappings_capabilities: pd.Series with index values:
            _source: bool
                Is this field name a top-level source field?
            ed_dtype: str
                The Elasticsearch data type
            pd_dtype: str
                The pandas data type
            searchable: bool
                Is the field searchable in Elasticsearch?
            aggregatable: bool
                Is the field aggregatable in Elasticsearch?
        """
        try:
            field_capabilities = self._mappings_capabilities.loc[field_name]
        except KeyError:
            field_capabilities = pd.Series()
        return field_capabilities

    def get_date_field_format(self, field_name):
        """
        Parameters
        ----------
        field_name: str

        Returns
        -------
        dict
            A dictionary (for date fields) containing the mapping {field_name:format}
        """
        return self._date_fields_format.get(field_name)

    def source_field_pd_dtype(self, field_name):
        """
        Parameters
        ----------
        field_name: str

        Returns
        -------
        is_source_field: bool
            Is this field name a top-level source field?
        pd_dtype: str
            The pandas data type we map to
        """
        pd_dtype = 'object'
        is_source_field = False

        if field_name in self._source_field_pd_dtypes:
            is_source_field = True
            pd_dtype = self._source_field_pd_dtypes[field_name]

        return is_source_field, pd_dtype

    def is_source_field(self, field_name):
        """
        Parameters
        ----------
        field_name: str

        Returns
        -------
        is_source_field: bool
            Is this field name a top-level source field?
        """
        is_source_field = False

        if field_name in self._source_field_pd_dtypes:
            is_source_field = True

        return is_source_field

    def aggregatable_field_names(self, field_names=None):
        """
        Return a dict of aggregatable field_names from all field_names or field_names list
        {'customer_full_name': 'customer_full_name.keyword', ...}

        Logic here is that field_name names are '_source' fields and keyword fields
        may be nested beneath the field. E.g.
        customer_full_name: text
        customer_full_name.keyword: keyword

        customer_full_name.keyword is the aggregatable field for customer_full_name

        Returns
        -------
        dict
            e.g. {'customer_full_name': 'customer_full_name.keyword', ...}
        """
        if field_names is None:
            field_names = self.source_fields()
        aggregatables = {}
        for field_name in field_names:
            capabilities = self.field_capabilities(field_name)
            if capabilities['aggregatable']:
                aggregatables[field_name] = field_name
            else:
                # Try 'field_name.keyword'
                field_name_keyword = field_name + '.keyword'
                capabilities = self.field_capabilities(field_name_keyword)
                if not capabilities.empty and capabilities.get('aggregatable'):
                    aggregatables[field_name_keyword] = field_name

        if not aggregatables:
            raise ValueError("Aggregations not supported for ", field_name)

        return aggregatables

    def numeric_source_fields(self, field_names, include_bool=True):
        """
        Returns
        -------
        numeric_source_fields: list of str
            List of source fields where pd_dtype == (int64 or float64 or bool)
        """
        if include_bool:
            df = self._mappings_capabilities[self._mappings_capabilities._source &
                                             ((self._mappings_capabilities.pd_dtype == 'int64') |
                                              (self._mappings_capabilities.pd_dtype == 'float64') |
                                              (self._mappings_capabilities.pd_dtype == 'bool'))]
        else:
            df = self._mappings_capabilities[self._mappings_capabilities._source &
                                             ((self._mappings_capabilities.pd_dtype == 'int64') |
                                              (self._mappings_capabilities.pd_dtype == 'float64'))]
        # if field_names exists, filter index with field_names
        if field_names is not None:
            # reindex adds NA for non-existing field_names (non-numeric), so drop these after reindex
            df = df.reindex(field_names)
            df.dropna(inplace=True)

        # return as list
        return df.index.to_list()

    def source_fields(self):
        """
        Returns
        -------
        source_fields: list of str
            List of source fields
        """
        return self._source_field_pd_dtypes.keys()

    def count_source_fields(self):
        """
        Returns
        -------
        count_source_fields: int
            Number of source fields in mapping
        """
        return len(self._source_field_pd_dtypes)

    def dtypes(self, field_names=None):
        """
        Returns
        -------
        dtypes: pd.Series
            Source field name + pd_dtype as np.dtype
        """
        if field_names is not None:
            return pd.Series(
                {key: np.dtype(self._source_field_pd_dtypes[key]) for key in field_names})

        return pd.Series(
            {key: np.dtype(value) for key, value in self._source_field_pd_dtypes.items()})

    def info_es(self, buf):
        buf.write("Mappings:\n")
        buf.write(" capabilities: {0}\n".format(self._mappings_capabilities.to_string()))
        buf.write(" date_fields_format: {0}\n".format(self._date_fields_format))
