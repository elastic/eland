# Licensed to Elasticsearch B.V under one or more agreements.
# Elasticsearch B.V licenses this file to you under the Apache 2.0 License.
# See the LICENSE file in the project root for more information

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
        ("extended_stats", "count"),
        "cardinality",
        ("percentiles", "50.0"),
    ]


def test_extended_stats_optimization():
    # Tests that when '<agg>' and an 'extended_stats' agg are used together
    # that ('extended_stats', '<agg>') is used instead of '<agg>'.
    es_aggs = Operations._map_pd_aggs_to_es_aggs(["count", "nunique"])
    assert es_aggs == ["count", "cardinality"]

    for pd_agg in ["var", "std"]:
        extended_es_agg = Operations._map_pd_aggs_to_es_aggs([pd_agg])[0]

        es_aggs = Operations._map_pd_aggs_to_es_aggs([pd_agg, "nunique"])
        assert es_aggs == [extended_es_agg, "cardinality"]

        es_aggs = Operations._map_pd_aggs_to_es_aggs(["count", pd_agg, "nunique"])
        assert es_aggs == [("extended_stats", "count"), extended_es_agg, "cardinality"]
