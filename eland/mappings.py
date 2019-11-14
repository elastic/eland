import warnings

import pandas as pd
from pandas.core.dtypes.common import (is_float_dtype, is_bool_dtype, is_integer_dtype, is_datetime_or_timedelta_dtype,
                                       is_string_dtype)


class Mappings:
    """
    General purpose to manage Elasticsearch to/from pandas mappings

    Attributes
    ----------

    mappings_capabilities: pandas.DataFrame
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
        if (client is not None) and (index_pattern is not None):
            get_mapping = client.get_mapping(index=index_pattern)

            # Get all fields (including all nested) and then field_caps
            # for these names (fields=* doesn't appear to work effectively...)
            all_fields = Mappings._extract_fields_from_mapping(get_mapping)
            all_fields_caps = client.field_caps(index=index_pattern, fields=list(all_fields.keys()))

            # Get top level (not sub-field multifield) mappings
            source_fields = Mappings._extract_fields_from_mapping(get_mapping, source_only=True)

            # Populate capability matrix of fields
            # field_name, es_dtype, pd_dtype, is_searchable, is_aggregtable, is_source
            self._mappings_capabilities = Mappings._create_capability_matrix(all_fields, source_fields, all_fields_caps)
        else:
            # straight copy
            self._mappings_capabilities = mappings._mappings_capabilities.copy()

        # Cache source field types for efficient lookup
        # (this massively improves performance of DataFrame.flatten)
        self._source_field_pd_dtypes = {}

        for field_name in self._mappings_capabilities[self._mappings_capabilities._source == True].index:
            pd_dtype = self._mappings_capabilities.loc[field_name]['pd_dtype']
            self._source_field_pd_dtypes[field_name] = pd_dtype

    @staticmethod
    def _extract_fields_from_mapping(mappings, source_only=False):
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
        fields: dict
            Dict of field names and types

        """
        fields = {}

        # Recurse until we get a 'type: xxx'
        def flatten(x, name=''):
            if type(x) is dict:
                for a in x:
                    if a == 'type' and type(x[a]) is str:  # 'type' can be a name of a field
                        field_name = name[:-1]
                        field_type = x[a]

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

        columns = ['_source', 'es_dtype', 'pd_dtype', 'searchable', 'aggregatable']
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

        capability_matrix_df = pd.DataFrame.from_dict(capability_matrix, orient='index', columns=columns)

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

        mappings = {}
        mappings['properties'] = {}
        for column_name, dtype in dataframe.dtypes.iteritems():
            if geo_points is not None and column_name in geo_points:
                es_dtype = 'geo_point'
            else:
                es_dtype = Mappings._pd_dtype_to_es_dtype(dtype)

            mappings['properties'][column_name] = {}
            mappings['properties'][column_name]['type'] = es_dtype

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
        return self._mappings_capabilities.loc[field_name]

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

    def aggregatable_columns(self, columns=None):
        """
        Return a dict of aggregatable columns from all columns or columns list
        {'customer_full_name': 'customer_full_name.keyword', ...}

        Logic here is that column names are '_source' fields and keyword fields
        may be nested beneath the field. E.g.
        customer_full_name: text
        customer_full_name.keyword: keyword

        customer_full_name.keyword is the aggregatable field for customer_full_name

        Returns
        -------
        dict
            e.g. {'customer_full_name': 'customer_full_name.keyword', ...}
        """
        if columns is None:
            columns = self.source_fields()

        aggregatables = {}

        for column in columns:
            capabilities = self.field_capabilities(column)
            if capabilities['aggregatable']:
                aggregatables[column] = column
            else:
                # Try 'column.keyword'
                column_keyword = column + '.keyword'
                capabilities = self.field_capabilities(column_keyword)
                if capabilities['aggregatable']:
                    aggregatables[column_keyword] = column
                else:
                    # Aggregations not supported for this field
                    raise ValueError("Aggregations not supported for ", column)

        return aggregatables

    def numeric_source_fields(self, columns, include_bool=True):
        """
        Returns
        -------
        numeric_source_fields: list of str
            List of source fields where pd_dtype == (int64 or float64 or bool)
        """
        if columns is not None:
            if include_bool == True:
                return self._mappings_capabilities[(self._mappings_capabilities._source == True) &
                                                   ((self._mappings_capabilities.pd_dtype == 'int64') |
                                                    (self._mappings_capabilities.pd_dtype == 'float64') |
                                                    (self._mappings_capabilities.pd_dtype == 'bool'))].reindex(
                    columns).index.tolist()
            else:
                return self._mappings_capabilities[(self._mappings_capabilities._source == True) &
                                                   ((self._mappings_capabilities.pd_dtype == 'int64') |
                                                    (self._mappings_capabilities.pd_dtype == 'float64'))].reindex(
                    columns).index.tolist()
        else:
            if include_bool == True:
                return self._mappings_capabilities[(self._mappings_capabilities._source == True) &
                                                   ((self._mappings_capabilities.pd_dtype == 'int64') |
                                                    (self._mappings_capabilities.pd_dtype == 'float64') |
                                                    (self._mappings_capabilities.pd_dtype == 'bool'))].index.tolist()
            else:
                return self._mappings_capabilities[(self._mappings_capabilities._source == True) &
                                                   ((self._mappings_capabilities.pd_dtype == 'int64') |
                                                    (self._mappings_capabilities.pd_dtype == 'float64'))].index.tolist()

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
        return len(self.source_fields())

    def dtypes(self, columns=None):
        """
        Returns
        -------
        dtypes: pd.Series
            Source field name + pd_dtype
        """
        if columns is not None:
            return pd.Series(
                {key: self._source_field_pd_dtypes[key] for key in columns})

        return pd.Series(self._source_field_pd_dtypes)

    def info_es(self, buf):
        buf.write("Mappings:\n")
        buf.write(" capabilities: {0}\n".format(self._mappings_capabilities))
