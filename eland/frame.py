"""
DataFrame
---------
An efficient 2D container for potentially mixed-type time series or other
labeled data series.

The underlying data resides in Elasticsearch and the API aligns as much as
possible with pandas.DataFrame API.

This allows the eland.DataFrame to access large datasets stored in Elasticsearch,
without storing the dataset in local memory.

Implementation Details
----------------------

Elasticsearch indexes can be configured in many different ways, and these indexes
utilise different data structures to pandas.DataFrame.

eland.DataFrame operations that return individual rows (e.g. df.head()) return
_source data. If _source is not enabled, this data is not accessible.

Similarly, only Elasticsearch searchable fields can be searched or filtered, and
only Elasticsearch aggregatable fields can be aggregated or grouped.

"""
import eland

from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search

import pandas as pd

class DataFrame():
    """
    pandas.DataFrame like API that proxies into Elasticsearch index(es).

    Parameters
    ----------
    client : eland.Client
        A reference to a Elasticsearch python client

    index_pattern : str
        An Elasticsearch index pattern. This can contain wildcards (e.g. filebeat-*).

    See Also
    --------

    Examples
    --------

    >>> import eland as ed
    >>> client = ed.Client(Elasticsearch())
    >>> df = ed.DataFrame(client, 'reviews')
    >>> df.head()
       reviewerId  vendorId  rating              date
    0           0         0       5  2006-04-07 17:08
    1           1         1       5  2006-05-04 12:16
    2           2         2       4  2006-04-21 12:26
    3           3         3       5  2006-04-18 15:48
    4           3         4       5  2006-04-18 15:49

    Notice that the types are based on Elasticsearch mappings

    Notes
    -----
    If the Elasticsearch index is deleted or index mappings are changed after this
    object is created, the object is not rebuilt and so inconsistencies can occur.

    Mapping Elasticsearch types to pandas dtypes
    --------------------------------------------

    Elasticsearch field datatype              | Pandas dtype
    --
    text                                      | object
    keyword                                   | object
    long, integer, short, byte, binary        | int64
    double, float, half_float, scaled_float   | float64
    date, date_nanos                          | datetime64[ns]
    boolean                                   | bool
    TODO - add additional mapping types
    """
    def __init__(self, client, index_pattern):
        self.client = eland.Client(client)
        self.index_pattern = index_pattern

        # Get and persist mappings, this allows use to correctly
        # map returned types from Elasticsearch to pandas datatypes
        mapping = self.client.indices().get_mapping(index=self.index_pattern)
        #field_caps = self.client.field_caps(index=self.index_pattern, fields='*')

        #self.fields, self.aggregatable_fields, self.searchable_fields = \
        #    DataFrame._es_mappings_to_pandas(mapping, field_caps)
        self.fields = DataFrame._es_mappings_to_pandas(mapping)

    @staticmethod
    def _flatten_results(prefix, results, result):
        # TODO
        return prefix

    def _es_results_to_pandas(self, results):
        # TODO - resolve nested fields
        rows = []
        for hit in results['hits']['hits']:
            row = hit['_source']
            rows.append(row)

        df = pd.DataFrame(data=rows)

        return df

    @staticmethod
    def _extract_types_from_mapping(y):
        """
        Extract data types from mapping for DataFrame columns.

        Elasticsearch _source data is transformed into pandas DataFrames. This strategy is not compatible
        with all Elasticsearch configurations. Notes:

        - This strategy is not compatible with all Elasticsearch configurations. If _source is disabled
        (https://www.elastic.co/guide/en/elasticsearch/reference/current/mapping-source-field.html#disable-source-field)
        no data values will be populated
        - Sub-fields (e.g. english.text in {"mappings":{"properties":{"text":{"type":"text","fields":{"english":{"type":"text","analyzer":"english"}}}}}})
        are not be used
        """
        out = {}

        # Recurse until we get a 'type: xxx' - ignore sub-fields
        def flatten(x, name=''):
            if type(x) is dict:
                for a in x:
                    if a == 'type' and type(x[a]) is str: # 'type' can be a name of a field
                        out[name[:-1]] = x[a]
                    if a == 'properties' or a == 'fields':
                        flatten(x[a], name)
                    else:
                        flatten(x[a], name + a + '.')

        flatten(y)

        return out

    @staticmethod
    def _es_mappings_to_pandas(mappings):
        fields = {}
        for index in mappings:
            if 'properties' in mappings[index]['mappings']:
                properties = mappings[index]['mappings']['properties']
                
                datatypes = DataFrame._extract_types_from_mapping(properties)

                # Note there could be conflicts here - e.g. the same field name with different semantics in
                # different indexes - currently the last one wins TODO: review this
                fields.update(datatypes)

        return pd.DataFrame.from_dict(data=fields, orient='index', columns=['datatype'])

    def head(self, n=5):
        results = self.client.search(index=self.index_pattern, size=n)

        return self._es_results_to_pandas(results)
    
    def describe(self):
        # First get all types
        #mapping = self.client.indices().get_mapping(index=self.index_pattern)
        mapping = self.client.indices().get_mapping(index=self.index_pattern)

        fields = DataFrame._es_mappings_to_pandas(mapping)
        
        # Get numeric types (https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#the-where-method-and-masking)
        # https://www.elastic.co/guide/en/elasticsearch/reference/current/number.html
        # TODO refactor this list out of method
        numeric_fields = fields.query('datatype == ["long", "integer", "short", "byte", "double", "float", "half_float", "scaled_float"]')
                
        # for each field we copute:
        # count, mean, std, min, 25%, 50%, 75%, max
        search = search(using=self.client, index=self.index_pattern).extra(size=0)

        for field in numeric_fields.field:
            search.aggs.metric('extended_stats_'+field, 'extended_stats', field=field)
            search.aggs.metric('percentiles_'+field, 'percentiles', field=field)

        response = search.execute()

        results = pd.dataframe(index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])
        
        for field in numeric_fields.field:
            values = []
            values.append(response.aggregations['extended_stats_'+field]['count'])
            values.append(response.aggregations['extended_stats_'+field]['avg'])
            values.append(response.aggregations['extended_stats_'+field]['std_deviation'])
            values.append(response.aggregations['extended_stats_'+field]['min'])
            values.append(response.aggregations['percentiles_'+field]['values']['25.0'])
            values.append(response.aggregations['percentiles_'+field]['values']['50.0'])
            values.append(response.aggregations['percentiles_'+field]['values']['75.0'])
            values.append(response.aggregations['extended_stats_'+field]['max'])
            
            # if not None
            if (values.count(None) < len(values)):
                results = results.assign(**{field: values})
            
        return results
