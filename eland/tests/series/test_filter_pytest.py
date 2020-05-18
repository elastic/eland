# Licensed to Elasticsearch B.V under one or more agreements.
# Elasticsearch B.V licenses this file to you under the Apache 2.0 License.
# See the LICENSE file in the project root for more information

# File called _pytest for PyCharm compatability

import pytest
from eland.tests.common import TestData
from eland.tests.common import assert_pandas_eland_series_equal


class TestSeriesFilter(TestData):
    def test_filter_arguments_mutually_exclusive(self):
        ed_flights_small = self.ed_flights_small()["FlightDelayType"]

        with pytest.raises(TypeError):
            ed_flights_small.filter(items=[], like="!", regex="!")
        with pytest.raises(TypeError):
            ed_flights_small.filter(items=[], regex="!")
        with pytest.raises(TypeError):
            ed_flights_small.filter(items=[], like="!")
        with pytest.raises(TypeError):
            ed_flights_small.filter(like="!", regex="!")
        with pytest.raises(TypeError):
            ed_flights_small.filter()

    def test_filter_columns_not_allowed_for_series(self):
        ed_flights_small = self.ed_flights_small()["FlightDelayType"]
        pd_flights_small = self.pd_flights_small()["FlightDelayType"]

        with pytest.raises(ValueError):
            ed_flights_small.filter(regex=".*", axis="columns")
        with pytest.raises(ValueError):
            ed_flights_small.filter(regex=".*", axis=1)
        with pytest.raises(ValueError):
            pd_flights_small.filter(regex=".*", axis="columns")
        with pytest.raises(ValueError):
            pd_flights_small.filter(regex=".*", axis=1)

    @pytest.mark.parametrize("items", [[], ["20"], [str(x) for x in range(30)]])
    def test_flights_filter_index_items(self, items):
        ed_flights_small = self.ed_flights_small()["FlightDelayType"]
        pd_flights_small = self.pd_flights_small()["FlightDelayType"]

        ed_ser = ed_flights_small.filter(items=items, axis=0)
        pd_ser = pd_flights_small.filter(items=items, axis=0)

        assert_pandas_eland_series_equal(pd_ser, ed_ser)

    def test_flights_filter_index_like_and_regex(self):
        ed_flights_small = self.ed_flights_small()["FlightDelayType"]

        with pytest.raises(NotImplementedError):
            ed_flights_small.filter(like="2", axis=0)
        with pytest.raises(NotImplementedError):
            ed_flights_small.filter(regex="^2", axis=0)
