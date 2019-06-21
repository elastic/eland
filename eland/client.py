from elasticsearch import Elasticsearch

class Client():
    """
    eland client - implemented as facade to control access to Elasticsearch methods
    """
    def __init__(self, es=None):
        if isinstance(es, Elasticsearch):
            self.es = es
        else:
            self.es = Elasticsearch(es)

    def info(self):
        return self.es.info()
    
    def indices(self):
        return self.es.indices
    
    def search(self, **kwargs):
        return self.es.search(**kwargs)

    def field_caps(self, **kwargs):
        return self.es.field_caps(**kwargs)
