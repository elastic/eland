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

# File called _pytest for PyCharm compatability

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_index_equal, assert_series_equal

from tests.common import TestData


class TestGroupbyDataFrame(TestData):
    funcs = ["max", "min", "mean", "sum"]
    filter_data = [
        "AvgTicketPrice",
        "Cancelled",
        "dayOfWeek",
    ]
    ecommerce_filter_data = [
        "total_quantity",
        "geoip.region_name",
        "day_of_week",
        "total_unique_products",
        "taxful_total_price",
    ]

    @pytest.mark.parametrize("dropna", [True, False])
    @pytest.mark.parametrize("numeric_only", [True])
    def test_groupby_aggregate(self, numeric_only, dropna):
        # TODO Add tests for numeric_only=False for aggs
        # when we support aggregations on text fields
        pd_flights = self.pd_flights().filter(self.filter_data)
        ed_flights = self.ed_flights().filter(self.filter_data)

        pd_groupby = pd_flights.groupby("Cancelled", dropna=dropna).agg(
            self.funcs, numeric_only=numeric_only
        )
        ed_groupby = ed_flights.groupby("Cancelled", dropna=dropna).agg(
            self.funcs, numeric_only=numeric_only
        )

        # checking only values because dtypes are checked in aggs tests
        assert_frame_equal(pd_groupby, ed_groupby, check_exact=False, check_dtype=False)

    @pytest.mark.parametrize("pd_agg", funcs)
    def test_groupby_aggregate_single_aggs(self, pd_agg):
        pd_flights = self.pd_flights().filter(self.filter_data)
        ed_flights = self.ed_flights().filter(self.filter_data)

        pd_groupby = pd_flights.groupby("Cancelled").agg([pd_agg], numeric_only=True)
        ed_groupby = ed_flights.groupby("Cancelled").agg([pd_agg], numeric_only=True)

        # checking only values because dtypes are checked in aggs tests
        assert_frame_equal(pd_groupby, ed_groupby, check_exact=False, check_dtype=False)

    @pytest.mark.parametrize("dropna", [True, False])
    @pytest.mark.parametrize("pd_agg", ["max", "min", "mean", "sum", "median"])
    def test_groupby_aggs_numeric_only_true(self, pd_agg, dropna):
        # Pandas has numeric_only  applicable for the above aggs with groupby only.

        pd_flights = self.pd_flights().filter(self.filter_data)
        ed_flights = self.ed_flights().filter(self.filter_data)

        pd_groupby = getattr(pd_flights.groupby("Cancelled", dropna=dropna), pd_agg)(
            numeric_only=True
        )
        ed_groupby = getattr(ed_flights.groupby("Cancelled", dropna=dropna), pd_agg)(
            numeric_only=True
        )

        # checking only values because dtypes are checked in aggs tests
        assert_frame_equal(
            pd_groupby, ed_groupby, check_exact=False, check_dtype=False, rtol=2
        )

    @pytest.mark.parametrize("dropna", [True, False])
    @pytest.mark.parametrize("pd_agg", ["mad", "var", "std"])
    def test_groupby_aggs_mad_var_std(self, pd_agg, dropna):
        # For these aggs pandas doesn't support numeric_only
        pd_flights = self.pd_flights().filter(self.filter_data)
        ed_flights = self.ed_flights().filter(self.filter_data)

        pd_groupby = getattr(pd_flights.groupby("Cancelled", dropna=dropna), pd_agg)()
        ed_groupby = getattr(ed_flights.groupby("Cancelled", dropna=dropna), pd_agg)(
            numeric_only=True
        )

        # checking only values because dtypes are checked in aggs tests
        assert_frame_equal(
            pd_groupby, ed_groupby, check_exact=False, check_dtype=False, rtol=4
        )

    @pytest.mark.parametrize("pd_agg", ["nunique"])
    def test_groupby_aggs_nunique(self, pd_agg):
        pd_flights = self.pd_flights().filter(self.filter_data)
        ed_flights = self.ed_flights().filter(self.filter_data)

        pd_groupby = getattr(pd_flights.groupby("Cancelled"), pd_agg)()
        ed_groupby = getattr(ed_flights.groupby("Cancelled"), pd_agg)()

        # checking only values because dtypes are checked in aggs tests
        assert_frame_equal(
            pd_groupby, ed_groupby, check_exact=False, check_dtype=False, rtol=4
        )

    @pytest.mark.parametrize("pd_agg", ["max", "min", "mean", "median"])
    def test_groupby_aggs_numeric_only_false(self, pd_agg):
        pd_flights = self.pd_flights().filter(self.filter_data + ["timestamp"])
        ed_flights = self.ed_flights().filter(self.filter_data + ["timestamp"])

        # pandas numeric_only=False, matches with Eland numeric_only=None
        pd_groupby = getattr(pd_flights.groupby("Cancelled"), pd_agg)(
            numeric_only=False
        )
        ed_groupby = getattr(ed_flights.groupby("Cancelled"), pd_agg)(numeric_only=None)

        # sum usually returns NaT for Eland, Nothing is returned from pandas
        # we only check timestamp field here, because remaining cols are similar to numeric_only=True tests
        # assert_frame_equal doesn't work well for timestamp fields (It converts into int)
        # so we convert it into float
        pd_timestamp = pd.to_numeric(pd_groupby["timestamp"], downcast="float")
        ed_timestamp = pd.to_numeric(ed_groupby["timestamp"], downcast="float")

        assert_series_equal(pd_timestamp, ed_timestamp, check_exact=False, rtol=4)

    def test_groupby_columns(self):
        # Check errors
        ed_flights = self.ed_flights().filter(self.filter_data)

        match = "by parameter should be specified to groupby"
        with pytest.raises(ValueError, match=match):
            ed_flights.groupby(None).mean()

        by = ["ABC", "Cancelled"]
        match = "Requested columns 'ABC' not in the DataFrame"
        with pytest.raises(KeyError, match=match):
            ed_flights.groupby(by).mean()

    @pytest.mark.parametrize(
        "by",
        ["timestamp", "dayOfWeek", "Carrier", "Cancelled", ["dayOfWeek", "Carrier"]],
    )
    def test_groupby_different_dtypes(self, by):
        columns = ["dayOfWeek", "Carrier", "timestamp", "Cancelled"]
        pd_flights = self.pd_flights_small().filter(columns)
        ed_flights = self.ed_flights_small().filter(columns)

        pd_groupby = pd_flights.groupby(by).nunique()
        ed_groupby = ed_flights.groupby(by).nunique()

        assert list(pd_groupby.index) == list(ed_groupby.index)
        assert pd_groupby.index.dtype == ed_groupby.index.dtype
        assert list(pd_groupby.columns) == list(ed_groupby.columns)

    @pytest.mark.parametrize("dropna", [True, False])
    @pytest.mark.parametrize("groupby", ["geoip.region_name", "day_of_week"])
    @pytest.mark.parametrize("func", ["min", "max", "mean"])
    def test_groupby_dropna(self, dropna, func, groupby):
        pd_ecommerce = self.pd_ecommerce().filter(self.ecommerce_filter_data)
        ed_ecommerce = self.ed_ecommerce().filter(self.ecommerce_filter_data)

        pd_groupby = getattr(pd_ecommerce.groupby(groupby, dropna=dropna), func)(
            numeric_only=True
        )
        ed_groupby = getattr(ed_ecommerce.groupby(groupby, dropna=dropna), func)(
            numeric_only=True
        )

        assert_index_equal(pd_groupby.columns, ed_groupby.columns)
        assert_index_equal(pd_groupby.index, ed_groupby.index)
        assert_frame_equal(pd_groupby, ed_groupby, check_dtype=False)

    @pytest.mark.parametrize("groupby", ["dayOfWeek", ["dayOfWeek", "Cancelled"]])
    @pytest.mark.parametrize(
        ["func", "func_args"],
        [
            ("count", ()),
            ("agg", ("count",)),
            ("agg", (["count"],)),
            ("agg", (["max", "count", "min"],)),
        ],
    )
    def test_groupby_dataframe_count(self, groupby, func, func_args):
        pd_flights = self.pd_flights().filter(self.filter_data)
        ed_flights = self.ed_flights().filter(self.filter_data)

        pd_count = getattr(pd_flights.groupby(groupby), func)(*func_args)
        ed_count = getattr(ed_flights.groupby(groupby), func)(*func_args)

        assert_index_equal(pd_count.columns, ed_count.columns)
        assert_index_equal(pd_count.index, ed_count.index)
        assert_frame_equal(pd_count, ed_count)
        assert_series_equal(pd_count.dtypes, ed_count.dtypes)

    def test_groupby_dataframe_mad(self):
        pd_flights = self.pd_flights().filter(self.filter_data + ["DestCountry"])
        ed_flights = self.ed_flights().filter(self.filter_data + ["DestCountry"])

        pd_mad = pd_flights.groupby("DestCountry").mad()
        ed_mad = ed_flights.groupby("DestCountry").mad()

        assert_index_equal(pd_mad.columns, ed_mad.columns)
        assert_index_equal(pd_mad.index, ed_mad.index)
        assert_series_equal(pd_mad.dtypes, ed_mad.dtypes)

        pd_min_mad = pd_flights.groupby("DestCountry").aggregate(["min", "mad"])
        ed_min_mad = ed_flights.groupby("DestCountry").aggregate(["min", "mad"])

        assert_index_equal(pd_min_mad.columns, ed_min_mad.columns)
        assert_index_equal(pd_min_mad.index, ed_min_mad.index)
        assert_series_equal(pd_min_mad.dtypes, ed_min_mad.dtypes)

    def test_groupby_mode(self):
        ed_flights = self.ed_flights()
        match = "Currently mode is not supported for groupby"
        with pytest.raises(NotImplementedError, match=match):
            ed_flights.groupby("Cancelled").mode()

    @pytest.mark.parametrize("dropna", [True, False])
    @pytest.mark.parametrize(
        ["func", "args"],
        [
            ("quantile", ()),
            ("quantile", (0.55,)),
            ("quantile", ([0.2, 0.4, 0.6, 0.8],)),
        ],
    )
    @pytest.mark.parametrize("columns", ["Cancelled", ["dayOfWeek", "Cancelled"]])
    def test_groupby_aggs_quantile(self, dropna, columns, func, args):
        # Pandas has numeric_only  applicable for the above aggs with groupby only.

        pd_flights = self.pd_flights().filter(self.filter_data)
        ed_flights = self.ed_flights().filter(self.filter_data)

        pd_groupby = getattr(pd_flights.groupby(columns, dropna=dropna), func)(*args)
        ed_groupby = getattr(ed_flights.groupby(columns, dropna=dropna), func)(*args)
        # checking only values because dtypes are checked in aggs tests
        assert_frame_equal(
            pd_groupby, ed_groupby, check_exact=False, check_dtype=False, rtol=2
        )
