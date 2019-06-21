import warnings
import pandas as pd

class Mappings():
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
    def __init__(self, client, index_pattern):
        """
        Parameters
        ----------
        client: eland.Client
            Elasticsearch client

        index_pattern: str
            Elasticsearch index pattern
        """
        # persist index_pattern for debugging
        self.index_pattern = index_pattern

        mappings = client.indices().get_mapping(index=index_pattern)

        # Get all fields (including all nested) and then field_caps
        # for these names (fields=* doesn't appear to work effectively...)
        all_fields = Mappings._extract_fields_from_mapping(mappings)
        all_fields_caps = client.field_caps(index=index_pattern, fields=list(all_fields.keys()))

        # Get top level (not sub-field multifield) mappings
        source_fields = Mappings._extract_fields_from_mapping(mappings, source_only=True)

        # Populate capability matrix of fields
        # field_name, es_dtype, pd_dtype, is_searchable, is_aggregtable, is_source
        self.mappings_capabilities = Mappings._create_capability_matrix(all_fields, source_fields, all_fields_caps)

        # Cache source field types for efficient lookup
        # (this massively improves performance of DataFrame.flatten)
        self.source_field_pd_dtypes = {}

        for field_name in source_fields:
            pd_dtype = self.mappings_capabilities.loc[field_name]['pd_dtype']
            self.source_field_pd_dtypes[field_name] = pd_dtype

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
                                      format(field_name, vv['non_aggregatable_indices']),
                                      UserWarning)
                    if 'non_searchable_indices' in vv:
                        warnings.warn("Field {} has conflicting searchable fields across indexes {}",
                                      format(field_name, vv['non_searchable_indices']),
                                      UserWarning)

        capability_matrix_df = pd.DataFrame.from_dict(capability_matrix, orient='index', columns=columns)

        return capability_matrix_df.sort_index()

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

    def all_fields(self):
        """
        Returns
        -------
        all_fields: list
            All typed fields in the index mapping
        """
        return self.mappings_capabilities.index.tolist()

    """
    def pd_dtypes_groupby_source_fields(self):
        Returns
        -------
        groups: dict
            Calls pandas.core.groupby.GroupBy.groups for _source fields
            E.g.
            {
                'bool': Index(['Cancelled', 'FlightDelay'], dtype='object'),
                'datetime64[ns]': Index(['timestamp'], dtype='object'),
                'float64': Index(['AvgTicketPrice', 'DistanceKilometers', 'DistanceMiles',...
            }
        return self.mappings_capabilities[self.mappings_capabilities._source == True].groupby('pd_dtype').groups

    def pd_dtype
    """

    def is_source_field(self, field_name):
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

        if field_name in self.source_field_pd_dtypes:
            is_source_field = True
            pd_dtype = self.source_field_pd_dtypes[field_name]

        return is_source_field, pd_dtype

    def numeric_source_fields(self):
        """
        Returns
        -------
        numeric_source_fields: list of str
            List of source fields where pd_dtype == (int64 or float64)
        """
        return self.mappings_capabilities[(self.mappings_capabilities._source == True) &
                                          ((self.mappings_capabilities.pd_dtype == 'int64') |
                                           (self.mappings_capabilities.pd_dtype == 'float64'))].index.tolist()

