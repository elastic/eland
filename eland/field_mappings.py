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
from collections import OrderedDict

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import (is_float_dtype, is_bool_dtype, is_integer_dtype, is_datetime_or_timedelta_dtype,
                                       is_string_dtype)
from pandas.core.dtypes.inference import is_list_like


class FieldMappings:
    """
    General purpose to manage Elasticsearch to/from pandas mappings

    Attributes
    ----------

    _mappings_capabilities: pandas.DataFrame
        A data frame summarising the capabilities of the index mapping

        index                       - the eland display name

        es_field_name               - the Elasticsearch field name
        is_source                   - is top level field (i.e. not a multi-field sub-field)
        es_dtype                    - Elasticsearch field datatype
        es_date_format              - Elasticsearch date format (or None)
        pd_dtype                    - Pandas datatype
        is_searchable               - is the field searchable?
        is_aggregatable             - is the field aggregatable?
        is_scripted                 - is the field a scripted_field?
        aggregatable_es_field_name  - either es_field_name (if aggregatable),
                                      or es_field_name.keyword (if exists) or None
    """

    # the labels for each column (display_name is index)
    column_labels = ['es_field_name', 'is_source', 'es_dtype', 'es_date_format', 'pd_dtype', 'is_searchable',
                     'is_aggregatable', 'is_scripted', 'aggregatable_es_field_name']

    def __init__(self,
                 client=None,
                 index_pattern=None,
                 display_names=None):
        """
        Parameters
        ----------
        client: eland.Client
            Elasticsearch client

        index_pattern: str
            Elasticsearch index pattern

        display_names: list of str
            Field names to display
        """
        if (client is None) or (index_pattern is None):
            raise ValueError("Can not initialise mapping without client or index_pattern {} {}", client, index_pattern)

        get_mapping = client.get_mapping(index=index_pattern)

        # Get all fields (including all nested) and then all field_caps
        all_fields = FieldMappings._extract_fields_from_mapping(get_mapping)
        all_fields_caps = client.field_caps(index=index_pattern, fields='*')

        # Get top level (not sub-field multifield) mappings
        source_fields = FieldMappings._extract_fields_from_mapping(get_mapping, source_only=True)

        # Populate capability matrix of fields
        self._mappings_capabilities = FieldMappings._create_capability_matrix(all_fields, source_fields,
                                                                              all_fields_caps)

        if display_names is not None:
            self.display_names = display_names

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
        fields, dates_format: tuple(OrderedDict, dict)
            where:
                fields: OrderedDict of field names and types
                dates_format: Dict of date field names and format

        """
        fields = OrderedDict()

        # Recurse until we get a 'type: xxx'
        def flatten(x, name=''):
            if type(x) is dict:
                for a in x:
                    if a == 'type' and type(x[a]) is str:  # 'type' can be a name of a field
                        field_name = name[:-1]
                        field_type = x[a]
                        # if field_type is 'date' keep track of the format info when available
                        date_format = None
                        if field_type == "date" and "format" in x:
                            date_format = x["format"]
                        # If there is a conflicting type, warn - first values added wins
                        if field_name in fields and fields[field_name] != field_type:
                            warnings.warn("Field {} has conflicting types {} != {}".
                                          format(field_name, fields[field_name], field_type),
                                          UserWarning)
                        else:
                            fields[field_name] = (field_type, date_format)
                    elif a == 'properties' or (not source_only and a == 'fields'):
                        flatten(x[a], name)
                    elif not (source_only and a == 'fields'):  # ignore multi-field fields for source_only
                        flatten(x[a], name + a + '.')

        for index in mappings:
            if 'properties' in mappings[index]['mappings']:
                properties = mappings[index]['mappings']['properties']

                flatten(properties)

        return fields

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

        capability_matrix = OrderedDict()

        for field, field_caps in all_fields_caps_fields.items():
            if field in all_fields:
                # v = {'long': {'type': 'long', 'searchable': True, 'aggregatable': True}}
                for kk, vv in field_caps.items():
                    _source = (field in source_fields)
                    es_field_name = field
                    es_dtype = vv['type']
                    es_date_format = all_fields[field][1]
                    pd_dtype = FieldMappings._es_dtype_to_pd_dtype(vv['type'])
                    is_searchable = vv['searchable']
                    is_aggregatable = vv['aggregatable']
                    scripted = False
                    aggregatable_es_field_name = None  # this is populated later

                    caps = [es_field_name, _source, es_dtype, es_date_format, pd_dtype, is_searchable, is_aggregatable,
                            scripted, aggregatable_es_field_name]

                    capability_matrix[field] = caps

                    if 'non_aggregatable_indices' in vv:
                        warnings.warn("Field {} has conflicting aggregatable fields across indexes {}",
                                      format(field, vv['non_aggregatable_indices']),
                                      UserWarning)
                    if 'non_searchable_indices' in vv:
                        warnings.warn("Field {} has conflicting searchable fields across indexes {}",
                                      format(field, vv['non_searchable_indices']),
                                      UserWarning)

        capability_matrix_df = pd.DataFrame.from_dict(capability_matrix, orient='index',
                                                      columns=FieldMappings.column_labels)

        def find_aggregatable(row, df):
            # convert series to dict so we can add 'aggregatable_es_field_name'
            row_as_dict = row.to_dict()
            if row_as_dict['is_aggregatable'] == False:
                # if not aggregatable, then try field.keyword
                es_field_name_keyword = row.es_field_name + '.keyword'
                try:
                    series = df.loc[df.es_field_name == es_field_name_keyword]
                    if not series.empty and series.is_aggregatable.squeeze():
                        row_as_dict['aggregatable_es_field_name'] = es_field_name_keyword
                    else:
                        row_as_dict['aggregatable_es_field_name'] = None
                except KeyError:
                    row_as_dict['aggregatable_es_field_name'] = None
            else:
                row_as_dict['aggregatable_es_field_name'] = row_as_dict['es_field_name']

            return pd.Series(data=row_as_dict)

        # add aggregatable_es_field_name column by applying action to each row
        capability_matrix_df = capability_matrix_df.apply(find_aggregatable, args=(capability_matrix_df,),
                                                          axis='columns')

        # return just source fields (as these are the only ones we display)
        return capability_matrix_df[capability_matrix_df.is_source].sort_index()

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
                es_dtype = FieldMappings._pd_dtype_to_es_dtype(dtype)

            mappings['properties'][field_name_name] = OrderedDict()
            mappings['properties'][field_name_name]['type'] = es_dtype

        return {"mappings": mappings}

    def aggregatable_field_name(self, display_name):
        """
        Return a single aggregatable field_name from display_name

        Logic here is that field_name names are '_source' fields and keyword fields
        may be nested beneath the field. E.g.
        customer_full_name: text
        customer_full_name.keyword: keyword

        customer_full_name.keyword is the aggregatable field for customer_full_name

        Parameters
        ----------
        display_name: str

        Returns
        -------
        aggregatable_es_field_name: str or None
            The aggregatable field name associated with display_name. This could be the field_name, or the
            field_name.keyword.

        raise KeyError if the field_name doesn't exist in the mapping, or isn't aggregatable
        """
        if display_name not in self._mappings_capabilities.index:
            raise KeyError("Can not get aggregatable field name for invalid display name {}".format(display_name))

        if self._mappings_capabilities.loc[display_name].aggregatable_es_field_name is None:
            warnings.warn("Aggregations not supported for '{}' '{}'".format(display_name,
                                                                            self._mappings_capabilities.loc[
                                                                                display_name].es_field_name))

        return self._mappings_capabilities.loc[display_name].aggregatable_es_field_name

    def aggregatable_field_names(self):
        """
        Return a list of aggregatable Elasticsearch field_names for all display names.
        If field is not aggregatable_field_names, return nothing.

        Logic here is that field_name names are '_source' fields and keyword fields
        may be nested beneath the field. E.g.
        customer_full_name: text
        customer_full_name.keyword: keyword

        customer_full_name.keyword is the aggregatable field for customer_full_name

        Returns
        -------
        Dict of aggregatable_field_names
            key = aggregatable_field_name, value = field_name
            e.g. {'customer_full_name.keyword': 'customer_full_name', ...}
        """
        non_aggregatables = self._mappings_capabilities[self._mappings_capabilities.aggregatable_es_field_name.isna()]
        if not non_aggregatables.empty:
            warnings.warn("Aggregations not supported for '{}'".format(non_aggregatables))

        aggregatables = self._mappings_capabilities[self._mappings_capabilities.aggregatable_es_field_name.notna()]

        # extract relevant fields and convert to dict
        # <class 'dict'>: {'category.keyword': 'category', 'currency': 'currency', ...
        return OrderedDict(
            aggregatables[['aggregatable_es_field_name', 'es_field_name']].to_dict(orient='split')['data'])

    def date_field_format(self, es_field_name):
        """
        Parameters
        ----------
        es_field_name: str


        Returns
        -------
        str
            A string (for date fields) containing the date format for the field
        """
        return self._mappings_capabilities.loc[
            self._mappings_capabilities.es_field_name == es_field_name].es_date_format.squeeze()

    def field_name_pd_dtype(self, es_field_name):
        """
        Parameters
        ----------
        es_field_name: str

        Returns
        -------
        pd_dtype: str
            The pandas data type we map to

        Raises
        ------
        KeyError
            If es_field_name does not exist in mapping
        """
        if es_field_name not in self._mappings_capabilities.es_field_name:
            raise KeyError("es_field_name {} does not exist".format(es_field_name))

        pd_dtype = self._mappings_capabilities.loc[
            self._mappings_capabilities.es_field_name == es_field_name
            ].pd_dtype.squeeze()
        return pd_dtype

    def add_scripted_field(self, scripted_field_name, display_name, pd_dtype):
        # if this display name is used somewhere else, drop it
        if display_name in self._mappings_capabilities.index:
            self._mappings_capabilities = self._mappings_capabilities.drop(index=[display_name])

        # ['es_field_name', 'is_source', 'es_dtype', 'es_date_format', 'pd_dtype', 'is_searchable',
        # 'is_aggregatable', 'is_scripted', 'aggregatable_es_field_name']

        capabilities = {display_name: [scripted_field_name,
                                       False,
                                       self._pd_dtype_to_es_dtype(pd_dtype),
                                       None,
                                       pd_dtype,
                                       True,
                                       True,
                                       True,
                                       scripted_field_name]}

        capability_matrix_row = pd.DataFrame.from_dict(capabilities, orient='index',
                                                       columns=FieldMappings.column_labels)

        self._mappings_capabilities = self._mappings_capabilities.append(capability_matrix_row)

    def numeric_source_fields(self):
        pd_dtypes, es_field_names, es_date_formats = self.metric_source_fields()
        return es_field_names

    def metric_source_fields(self, include_bool=False, include_timestamp=False):
        """
        Returns
        -------
        pd_dtypes: list of np.dtype
            List of pd_dtypes for es_field_names
        es_field_names: list of str
            List of source fields where pd_dtype == (int64 or float64 or bool or timestamp)
        es_date_formats: list of str (can be None)
            List of es date formats for es_field

        TODO - not very efficient, but unless called per row, this should be ok
        """
        pd_dtypes = []
        es_field_names = []
        es_date_formats = []
        for index, row in self._mappings_capabilities.iterrows():
            pd_dtype = row['pd_dtype']
            es_field_name = row['es_field_name']
            es_date_format = row['es_date_format']

            if is_integer_dtype(pd_dtype) or is_float_dtype(pd_dtype):
                pd_dtypes.append(np.dtype(pd_dtype))
                es_field_names.append(es_field_name)
                es_date_formats.append(es_date_format)
            elif include_bool and is_bool_dtype(pd_dtype):
                pd_dtypes.append(np.dtype(pd_dtype))
                es_field_names.append(es_field_name)
                es_date_formats.append(es_date_format)
            elif include_timestamp and is_datetime_or_timedelta_dtype(pd_dtype):
                pd_dtypes.append(np.dtype(pd_dtype))
                es_field_names.append(es_field_name)
                es_date_formats.append(es_date_format)

        # return in display_name order
        return pd_dtypes, es_field_names, es_date_formats

    def get_field_names(self, include_scripted_fields=True):
        if include_scripted_fields:
            return self._mappings_capabilities.es_field_name.to_list()

        return self._mappings_capabilities[
            self._mappings_capabilities.is_scripted == False
            ].es_field_name.to_list()

    def _get_display_names(self):
        return self._mappings_capabilities.index.to_list()

    def _set_display_names(self, display_names):
        if not is_list_like(display_names):
            raise ValueError("'{}' is not list like".format(display_names))

        if list(set(display_names) - set(self.display_names)):
            raise KeyError("{} not in display names {}".format(display_names, self.display_names))

        self._mappings_capabilities = self._mappings_capabilities.reindex(display_names)

    display_names = property(_get_display_names, _set_display_names)

    def dtypes(self):
        """
        Returns
        -------
        dtypes: pd.Series
            Index: Display name
            Values: pd_dtype as np.dtype
        """
        pd_dtypes = self._mappings_capabilities['pd_dtype']

        # Set name of the returned series as None
        pd_dtypes.name = None

        # Convert return from 'str' to 'np.dtype'
        return pd_dtypes.apply(lambda x: np.dtype(x))

    def info_es(self, buf):
        buf.write("Mappings:\n")
        buf.write(" capabilities:\n{0}\n".format(self._mappings_capabilities.to_string()))

    def rename(self, old_name_new_name_dict):
        """
        Renames display names in-place

        Parameters
        ----------
        old_name_new_name_dict

        Returns
        -------
        Nothing

        Notes
        -----
        For the names that do not exist this is a no op
        """
        self._mappings_capabilities = self._mappings_capabilities.rename(index=old_name_new_name_dict)

    def get_renames(self):
        # return dict of renames { old_name: new_name, ... } (inefficient)
        renames = {}

        for display_name in self.display_names:
            field_name = self._mappings_capabilities.loc[display_name].es_field_name
            if field_name != display_name:
                renames[field_name] = display_name

        return renames
