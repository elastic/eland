#  Licensed to Elasticsearch B.V. under one or more contributor
#  license agreements. See the NOTICE file distributed with
#  this work for additional information regarding copyright
#  ownership. Elasticsearch B.V. licenses this file to you under
#  the Apache License, Version 2.0 (the "License"); you may
#  not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
# 	http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the License is distributed on an
#  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
#  specific language governing permissions and limitations
#  under the License.

import warnings
from copy import deepcopy
from typing import Any, Dict, List, Optional

from eland.filter import BooleanFilter, IsIn, IsNull, NotNull, RandomScoreFilter, Rlike


class Query:
    """
    Simple class to manage building Elasticsearch queries.
    """

    def __init__(self, query: Optional["Query"] = None):
        # type defs
        self._query: BooleanFilter
        self._aggs: Dict[str, Any]
        self._composite_aggs: Dict[str, Any]

        if query is None:
            self._query = BooleanFilter()
            self._aggs = {}
            self._composite_aggs = {}
        else:
            # Deep copy the incoming query so we can change it
            self._query = deepcopy(query._query)
            self._aggs = deepcopy(query._aggs)
            self._composite_aggs = deepcopy(query._composite_aggs)

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

    def terms_aggs(
        self,
        name: str,
        func: str,
        field: str,
        es_size: Optional[int] = None,
        missing: Optional[Any] = None,
    ) -> None:
        """
        Add terms agg e.g

        "aggs": {
            "name": {
                "terms": {
                    "field": "Airline",
                    "size": 10,
                    "missing": "null"
                }
            }
        }
        """
        agg = {func: {"field": field}}
        if es_size:
            agg[func]["size"] = str(es_size)

        if missing:
            agg[func]["missing"] = missing
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

    def percentile_agg(self, name: str, field: str, percents: List[float]) -> None:
        """

        Ref: https://www.elastic.co/guide/en/elasticsearch/reference/current/search-aggregations-metrics-percentile-aggregation.html

        "aggs": {
            "percentile_": {
                "percentiles": {
                    "field": "AvgTicketPrice",
                    "percents": [95, 99, 99.0]
                }
            }
        }

        """
        agg = {"percentiles": {"field": field, "percents": percents}}
        self._aggs[name] = agg

    def top_hits_agg(
        self,
        name: str,
        source_columns: List[str],
        sort_order: str,
        size: int = 1,
    ) -> None:

        top_hits: Any = {}
        if sort_order:
            top_hits["sort"] = [{i: {"order": sort_order}} for i in source_columns]
        if source_columns:
            top_hits["_source"] = {"includes": source_columns}
        top_hits["size"] = size
        self._aggs[name] = {"top_hits": top_hits}

    def composite_agg_bucket_terms(self, name: str, field: str) -> None:
        """
        Add terms agg for composite aggregation

        "aggs": {
            "name": {
                "terms": {
                    "field": "AvgTicketPrice"
                }
            }
        }
        """
        self._composite_aggs[name] = {"terms": {"field": field}}

    def composite_agg_bucket_date_histogram(
        self,
        name: str,
        field: str,
        calendar_interval: Optional[str] = None,
        fixed_interval: Optional[str] = None,
    ) -> None:
        if (calendar_interval is None) == (fixed_interval is None):
            raise ValueError(
                "calendar_interval and fixed_interval parmaeters are mutually exclusive"
            )
        agg = {"field": field}
        if calendar_interval is not None:
            agg["calendar_interval"] = calendar_interval
        elif fixed_interval is not None:
            agg["fixed_interval"] = fixed_interval
        self._composite_aggs[name] = {"date_histogram": agg}

    def composite_agg_start(
        self,
        name: str,
        size: int,
        dropna: bool = True,
    ) -> None:
        """
        Start a composite aggregation. This should be called
        after calls to composite_agg_bucket_*(), etc.

        https://www.elastic.co/guide/en/elasticsearch/reference/current/search-aggregations-bucket-composite-aggregation.html

        "aggs": {
            "groupby_buckets": {
                "composite": {
                    "size": 10,
                    "sources": [
                        {"total_quantity": {"terms": {"field": "total_quantity"}}}
                    ],
                    "after": {"total_quantity": 8},
                },
                "aggregations": {
                    "taxful_total_price_avg": {
                        "avg": {"field": "taxful_total_price"}
                    }
                },
            }
        }

        Parameters
        ----------
        size: int or None
            Use composite aggregation with pagination if size is not None
        name: str
            Name of the buckets
        dropna: bool
            Drop None values if True.
            TODO Not yet implemented

        """
        sources: List[Dict[str, Dict[str, str]]] = []

        # Go through all composite source aggregations
        # and apply dropna if needed.
        for bucket_agg_name, bucket_agg in self._composite_aggs.items():
            if bucket_agg.get("terms") and not dropna:
                bucket_agg = bucket_agg.copy()
                bucket_agg["terms"]["missing_bucket"] = "true"
            sources.append({bucket_agg_name: bucket_agg})
        self._composite_aggs.clear()

        aggs: Dict[str, Dict[str, Any]] = {
            "composite": {"size": size, "sources": sources}
        }

        if self._aggs:
            aggs["aggregations"] = self._aggs.copy()

        self._aggs.clear()
        self._aggs[name] = aggs

    def composite_agg_after_key(self, name: str, after_key: Dict[str, Any]) -> None:
        """
        Add's after_key to existing query to fetch next bunch of results

        Parameters
        ----------
        name: str
            Name of the buckets
        after_key: Dict[str, Any]
            Dictionary returned from previous query results
        """
        self._aggs[name]["composite"]["after"] = after_key

    def hist_aggs(
        self,
        name: str,
        field: str,
        min_value: Any,
        max_value: Any,
        num_bins: int,
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
