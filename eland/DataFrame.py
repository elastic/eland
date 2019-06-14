import eland

from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search

import json

import pandas as pd

class DataFrame():
    
    def __init__(self, client, index_pattern):
        self.client = eland.Client(client)
        self.index_pattern = index_pattern
        
        self.client.indices().exists(index_pattern)

    @staticmethod
    def _flatten_results(prefix, results, result):
        # TODO
        return prefix
        
    @staticmethod
    def _es_results_to_pandas(results):
        # TODO - resolve nested fields
        rows = []
        for hit in results['hits']['hits']:
            row = hit['_source']
            rows.append(row)
        #return pd.DataFrame(data=rows)
        # Converting the list of dicts to a dataframe doesn't convert datetimes
        # effectively compared to read_json. TODO - https://github.com/elastic/eland/issues/2
        json_rows = json.dumps(rows)
        return pd.read_json(json_rows)

    @staticmethod
    def _flatten_mapping(prefix, properties, result):
        for k, v in properties.items():
            if 'properties' in v:
                if(prefix == ''):
                    prefix = k
                else:
                    prefix = prefix + '.' + k
                DataFrame._flatten_mapping(prefix, v['properties'], result)
            else:
                if(prefix == ''):
                    key = k
                else:
                    key = prefix + '.' + k
                type = v['type']
                result.append((key, type))
    
    @staticmethod
    def _es_mappings_to_pandas(mappings):
        fields = []
        for index in mappings:            
            if 'properties' in mappings[index]['mappings']:
                properties = mappings[index]['mappings']['properties']
                
                DataFrame._flatten_mapping('', properties, fields)
                
        return pd.DataFrame(data=fields, columns=['field', 'datatype'])
        
    def head(self, n=5):
        results = self.client.search(index=self.index_pattern, size=n)

        return DataFrame._es_results_to_pandas(results)
    
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
        search = Search(using=self.client, index=self.index_pattern).extra(size=0)
        
        for field in numeric_fields.field:
            search.aggs.metric('extended_stats_'+field, 'extended_stats', field=field)
            search.aggs.metric('percentiles_'+field, 'percentiles', field=field)

        response = search.execute()
        
        results = pd.DataFrame(index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])
        
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
