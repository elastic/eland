import warnings
from copy import deepcopy

from eland.filter import BooleanFilter, NotNull, IsNull, IsIn

class Query:
    """
    Simple class to manage building Elasticsearch queries.

    Specifically, this

    """

    def __init__(self, query=None):
        if query is None:
            self._query = BooleanFilter()
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
            if self._query.empty():
                self._query = NotNull(field)
            else:
                self._query = self._query & NotNull(field)
        else:
            if self._query.empty():
                self._query = IsNull(field)
            else:
                self._query = self._query & IsNull(field)

    def ids(self, items, must=True):
        """
        Add ids query
        https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-ids-query.html
        """
        if must:
            if self._query.empty():
                self._query = IsIn('ids', items)
            else:
                self._query = self._query & IsIn('ids', items)
        else:
            if self._query.empty():
                self._query = ~(IsIn('ids', items))
            else:
                self._query = self._query & ~(IsIn('ids', items))

    def terms(self, field, items, must=True):
        """
        Add ids query
        https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-terms-query.html
        """
        if must:
            if self._query.empty():
                self._query = IsIn(field, items)
            else:
                self._query = self._query & IsIn(field, items)
        else:
            if self._query.empty():
                self._query = ~(IsIn(field, items))
            else:
                self._query = self._query & ~(IsIn(field, items))

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

    def hist_aggs(self, name, field, min_aggs, max_aggs, num_bins):
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
        min = min_aggs[field]
        max = max_aggs[field]

        interval = (max - min) / num_bins

        agg = {
            "histogram": {
                "field": field,
                "interval": interval
            }
        }
        self._aggs[name] = agg

    def to_search_body(self):
        if self._query.empty():
            body = {"aggs": self._aggs}
        else:
            body = {"query": self._query.build(), "aggs": self._aggs}
        return body

    def to_count_body(self):
        if len(self._aggs) > 0:
            warnings.warn('Requesting count for agg query {}', self)
        if self._query.empty():
            body = None
        else:
            body = {"query": self._query.build()}

        return body

    def update_boolean_filter(self, boolean_filter):
        if self._query.empty():
            self._query = boolean_filter
        else:
            self._query = self._query & boolean_filter

    def __repr__(self):
        return repr(self.to_search_body())
