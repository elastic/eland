# Licensed to Elasticsearch B.V under one or more agreements.
# Elasticsearch B.V licenses this file to you under the Apache 2.0 License.
# See the LICENSE file in the project root for more information

import warnings
from copy import deepcopy
from typing import Optional, Dict, List, Any

from eland.filter import (
    RandomScoreFilter,
    BooleanFilter,
    NotNull,
    IsNull,
    IsIn,
    Rlike,
)


class Query:
    """
    Simple class to manage building Elasticsearch queries.
    """

    def __init__(self, query: Optional["Query"] = None):
        # type defs
        self._query: BooleanFilter
        self._aggs: Dict[str, Any]

        if query is None:
            self._query = BooleanFilter()
            self._aggs = {}
        else:
            # Deep copy the incoming query so we can change it
            self._query = deepcopy(query._query)
            self._aggs = deepcopy(query._aggs)

    def exists(self, field: str, must: bool = True) -> None:
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

    def ids(self, items: List[Any], must: bool = True) -> None:
        """
        Add ids query
        https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-ids-query.html
        """
        if must:
            if self._query.empty():
                self._query = IsIn("ids", items)
            else:
                self._query = self._query & IsIn("ids", items)
        else:
            if self._query.empty():
                self._query = ~(IsIn("ids", items))
            else:
                self._query = self._query & ~(IsIn("ids", items))

    def terms(self, field: str, items: List[str], must: bool = True) -> None:
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

    def regexp(self, field: str, value: str) -> None:
        """
        Add regexp query
        https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-regexp-query.html
        """
        if self._query.empty():
            self._query = Rlike(field, value)
        else:
            self._query = self._query & Rlike(field, value)

    def terms_aggs(self, name: str, func: str, field: str, es_size: int) -> None:
        """
        Add terms agg e.g

        "aggs": {
            "name": {
                "terms": {
                    "field": "Airline",
                    "size": 10
                }
            }
        }
        """
        agg = {func: {"field": field, "size": es_size}}
        self._aggs[name] = agg

    def metric_aggs(self, name: str, func: str, field: str) -> None:
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
        agg = {func: {"field": field}}
        self._aggs[name] = agg

    def hist_aggs(
        self, name: str, field: str, min_value: Any, max_value: Any, num_bins: int,
    ) -> None:
        """
        Add histogram agg e.g.
        "aggs": {
            "name": {
                "histogram": {
                    "field": "AvgTicketPrice"
                    "interval": (max_value - min_value)/bins
                    "offset": min_value
                }
            }
        }
        """

        interval = (max_value - min_value) / num_bins

        if interval != 0:
            agg = {
                "histogram": {"field": field, "interval": interval, "offset": min_value}
            }
            self._aggs[name] = agg

    def to_search_body(self) -> Dict[str, Any]:
        body = {}
        if self._aggs:
            body["aggs"] = self._aggs
        if not self._query.empty():
            body["query"] = self._query.build()
        return body

    def to_count_body(self) -> Optional[Dict[str, Any]]:
        if len(self._aggs) > 0:
            warnings.warn(f"Requesting count for agg query {self}")
        if self._query.empty():
            return None
        else:
            return {"query": self._query.build()}

    def update_boolean_filter(self, boolean_filter: BooleanFilter) -> None:
        if self._query.empty():
            self._query = boolean_filter
        else:
            self._query = self._query & boolean_filter

    def random_score(self, random_state: int) -> None:
        self._query = RandomScoreFilter(self._query, random_state)

    def __repr__(self) -> str:
        return repr(self.to_search_body())
