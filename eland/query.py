import warnings
from copy import deepcopy


class Query:
    """
    Simple class to manage building Elasticsearch queries.

    Specifically, this

    """

    def __init__(self, query=None):
        if query is None:
            self._query = self._query_template()
            self._aggs = {}
        else:
            # Deep copy the incoming query so we can change it
            self._query = deepcopy(query._query)
            self._aggs = deepcopy(query._aggs)

    def exists(self, field, must=True):
        """
        Add exists query
        https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-exists-query.html
        """
        if must:
            self._query['bool']['must'].append({'exists': {'field': field}})
        else:
            self._query['bool']['must_not'].append({'exists': {'field': field}})

    def ids(self, items, must=True):
        """
        Add ids query
        https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-ids-query.html
        """
        if must:
            self._query['bool']['must'].append({'ids': {'values': items}})
        else:
            self._query['bool']['must_not'].append({'ids': {'values': items}})

    def terms(self, field, items, must=True):
        """
        Add ids query
        https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-terms-query.html
        """
        if must:
            self._query['bool']['must'].append({'terms': {field: items}})
        else:
            self._query['bool']['must_not'].append({'terms': {field: items}})

    def metric_aggs(self, name, func, field):
        """
        Add metric agg e.g

        "aggs": {
            "name": {
                "max": {
                    "field": "AvgTicketPrice"
                }
            }
        }
        """
        agg = {
            func: {
                "field": field
            }
        }
        self._aggs[name] = agg

    def hist_aggs(self, name, field, min_aggs, max_aggs, bins):
        """
        Add histogram agg e.g.
        "aggs": {
            "name": {
                "histogram": {
                    "field": "AvgTicketPrice"
                    "interval": (max_aggs[field] - min_aggs[field])/bins
                }
            }
        }
        """

    def to_search_body(self):
        body = {"query": self._query, "aggs": self._aggs}
        return body

    def to_count_body(self):
        if len(self._aggs) > 0:
            warnings.warn('Requesting count for agg query {}', self)
        body = {"query": self._query}

        return body

    def __repr__(self):
        return repr(self.to_search_body())

    @staticmethod
    def _query_template():
        template = {
            "bool": {
                "must": [],
                "must_not": []
            }
        }
        return deepcopy(template)
