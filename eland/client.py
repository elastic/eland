from elasticsearch import Elasticsearch
from elasticsearch import helpers

class Client():
    """
    eland client - implemented as facade to control access to Elasticsearch methods
    """
    def __init__(self, es=None):
        if isinstance(es, Elasticsearch):
            self.es = es
        elif isinstance(es, Client):
            self.es = es.es
        else:
            self.es = Elasticsearch(es)
            
    def info(self):
        return self.es.info()
    
    def indices(self):
        return self.es.indices

    def bulk(self, actions, refresh=False):
        return helpers.bulk(self.es, actions, refresh=refresh)

    def scan(self, **kwargs):
        return helpers.scan(self.es, **kwargs)

    def search(self, **kwargs):
        return self.es.search(**kwargs)

    def field_caps(self, **kwargs):
        return self.es.field_caps(**kwargs)

    def count(self, **kwargs):
        count_json = self.es.count(**kwargs)
        return count_json['count']

