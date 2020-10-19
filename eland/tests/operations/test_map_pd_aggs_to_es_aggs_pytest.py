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

from eland.operations import Operations


def test_all_aggs():
    es_aggs = Operations._map_pd_aggs_to_es_aggs(
        ["min", "max", "mean", "std", "var", "mad", "count", "nunique", "median"]
    )

    assert es_aggs == [
        ("extended_stats", "min"),
        ("extended_stats", "max"),
        ("extended_stats", "avg"),
        ("extended_stats", "std_deviation"),
        ("extended_stats", "variance"),
        "median_absolute_deviation",
        "value_count",
        "cardinality",
        ("percentiles", "50.0"),
    ]


def test_extended_stats_optimization():
    # Tests that when '<agg>' and an 'extended_stats' agg are used together
    # that ('extended_stats', '<agg>') is used instead of '<agg>'.
    es_aggs = Operations._map_pd_aggs_to_es_aggs(["count", "nunique"])
    assert es_aggs == ["value_count", "cardinality"]

    for pd_agg in ["var", "std"]:
        extended_es_agg = Operations._map_pd_aggs_to_es_aggs([pd_agg])[0]

        es_aggs = Operations._map_pd_aggs_to_es_aggs([pd_agg, "nunique"])
        assert es_aggs == [extended_es_agg, "cardinality"]

        es_aggs = Operations._map_pd_aggs_to_es_aggs(["count", pd_agg, "nunique"])
        assert es_aggs == ["value_count", extended_es_agg, "cardinality"]
