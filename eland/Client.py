from elasticsearch import Elasticsearch

# eland client - implement as facade to control access to Elasticsearch methods
class Client(object):

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
