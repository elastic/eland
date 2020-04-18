from eland.operations import Operations


def test_all_aggs():
    es_aggs = Operations._map_pd_aggs_to_es_aggs(
        ["min", "max", "mean", "std", "var", "mad", "count", "nunique", "median"]
    )

    assert es_aggs == [
        "min",
        "max",
        "avg",
        ("extended_stats", "std_deviation"),
        ("extended_stats", "variance"),
        "median_absolute_deviation",
        ("extended_stats", "count"),
        "cardinality",
        ("percentiles", "50.0"),
    ]


def test_extended_stats_count_optimization():
    # Tests that when 'count' and an 'extended_stats' agg are used together
    # that ('extended_stats', 'count') is used instead of 'count'.
    es_aggs = Operations._map_pd_aggs_to_es_aggs(["count", "mean"])
    assert es_aggs == ["count", "avg"]

    for pd_agg in ["var", "std"]:
        extended_es_agg = Operations._map_pd_aggs_to_es_aggs([pd_agg])[0]

        es_aggs = Operations._map_pd_aggs_to_es_aggs([pd_agg, "mean"])
        assert es_aggs == [extended_es_agg, "avg"]

        es_aggs = Operations._map_pd_aggs_to_es_aggs(["count", pd_agg, "mean"])
        assert es_aggs == [("extended_stats", "count"), extended_es_agg, "avg"]
