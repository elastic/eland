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

# File called _pytest for PyCharm compatibility

import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from eland.tests.common import TestData


class TestDataFrameMetrics(TestData):
    funcs = ["max", "min", "mean", "sum"]
    extended_funcs = ["median", "mad", "var", "std"]

    @pytest.mark.parametrize("numeric_only", [False, None])
    def test_flights_metrics(self, numeric_only):
        pd_flights = self.pd_flights()
        ed_flights = self.ed_flights()

        for func in self.funcs:
            # Pandas v1.0 doesn't support mean() on datetime
            # Pandas and Eland don't support sum() on datetime
            if not numeric_only:
                dtype_include = (
                    [np.number, np.datetime64]
                    if func not in ("mean", "sum")
                    else [np.number]
                )
                pd_flights = pd_flights.select_dtypes(include=dtype_include)
                ed_flights = ed_flights.select_dtypes(include=dtype_include)

            pd_metric = getattr(pd_flights, func)(numeric_only=numeric_only)
            ed_metric = getattr(ed_flights, func)(numeric_only=numeric_only)

            assert_series_equal(pd_metric, ed_metric)

    def test_flights_extended_metrics(self):
        pd_flights = self.pd_flights()
        ed_flights = self.ed_flights()

        # Test on reduced set of data for more consistent
        # median behaviour + better var, std test for sample vs population
        pd_flights = pd_flights[["AvgTicketPrice"]]
        ed_flights = ed_flights[["AvgTicketPrice"]]

        import logging

        logger = logging.getLogger("elasticsearch")
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.DEBUG)

        for func in self.extended_funcs:
            pd_metric = getattr(pd_flights, func)(
                **({"numeric_only": True} if func != "mad" else {})
            )
            ed_metric = getattr(ed_flights, func)(numeric_only=True)

            pd_value = pd_metric["AvgTicketPrice"]
            ed_value = ed_metric["AvgTicketPrice"]
            assert (ed_value * 0.9) <= pd_value <= (ed_value * 1.1)  # +/-10%

    def test_flights_extended_metrics_nan(self):
        pd_flights = self.pd_flights()
        ed_flights = self.ed_flights()

        # Test on single row to test NaN behaviour of sample std/variance
        pd_flights_1 = pd_flights[pd_flights.FlightNum == "9HY9SWR"][["AvgTicketPrice"]]
        ed_flights_1 = ed_flights[ed_flights.FlightNum == "9HY9SWR"][["AvgTicketPrice"]]

        for func in self.extended_funcs:
            pd_metric = getattr(pd_flights_1, func)()
            ed_metric = getattr(ed_flights_1, func)()

            assert_series_equal(
                pd_metric, ed_metric, check_exact=False, check_less_precise=True
            )

        # Test on zero rows to test NaN behaviour of sample std/variance
        pd_flights_0 = pd_flights[pd_flights.FlightNum == "XXX"][["AvgTicketPrice"]]
        ed_flights_0 = ed_flights[ed_flights.FlightNum == "XXX"][["AvgTicketPrice"]]

        for func in self.extended_funcs:
            pd_metric = getattr(pd_flights_0, func)()
            ed_metric = getattr(ed_flights_0, func)()

            assert_series_equal(
                pd_metric, ed_metric, check_exact=False, check_less_precise=True
            )

    def test_ecommerce_selected_non_numeric_source_fields(self):
        # None of these are numeric
        columns = [
            "category",
            "currency",
            "customer_birth_date",
            "customer_first_name",
            "user",
        ]

        pd_ecommerce = self.pd_ecommerce()[columns]
        ed_ecommerce = self.ed_ecommerce()[columns]

        for func in self.funcs:
            assert_series_equal(
                getattr(pd_ecommerce, func)(numeric_only=True),
                getattr(ed_ecommerce, func)(numeric_only=True),
                check_less_precise=True,
            )

    def test_ecommerce_selected_mixed_numeric_source_fields(self):
        # Some of these are numeric
        columns = [
            "category",
            "currency",
            "taxless_total_price",
            "customer_birth_date",
            "total_quantity",
            "customer_first_name",
            "user",
        ]

        pd_ecommerce = self.pd_ecommerce()[columns]
        ed_ecommerce = self.ed_ecommerce()[columns]

        for func in self.funcs:
            assert_series_equal(
                getattr(pd_ecommerce, func)(numeric_only=True),
                getattr(ed_ecommerce, func)(numeric_only=True),
                check_less_precise=True,
            )

    def test_ecommerce_selected_all_numeric_source_fields(self):
        # All of these are numeric
        columns = ["total_quantity", "taxful_total_price", "taxless_total_price"]

        pd_ecommerce = self.pd_ecommerce()[columns]
        ed_ecommerce = self.ed_ecommerce()[columns]

        for func in self.funcs:
            assert_series_equal(
                getattr(pd_ecommerce, func)(numeric_only=True),
                getattr(ed_ecommerce, func)(numeric_only=True),
                check_less_precise=True,
            )

    def test_flights_datetime_metrics_agg(self):
        ed_timestamps = self.ed_flights()[["timestamp"]]
        expected_values = {
            "timestamp": {
                "min": pd.Timestamp("2018-01-01 00:00:00"),
                "mean": pd.Timestamp("2018-01-21 19:20:45.564438232"),
                "max": pd.Timestamp("2018-02-11 23:50:12"),
                "nunique": 12236,
                "mad": pd.NaT,
                "std": pd.NaT,
                "sum": pd.NaT,
                "var": pd.NaT,
            }
        }

        ed_metrics = ed_timestamps.agg(self.funcs + self.extended_funcs + ["nunique"])
        ed_metrics_dict = ed_metrics.to_dict()
        ed_metrics_dict["timestamp"].pop("median")  # Median is tested below.
        assert ed_metrics_dict == expected_values

    @pytest.mark.parametrize("agg", ["mean", "min", "max", "nunique"])
    def test_flights_datetime_metrics_single_agg(self, agg):
        ed_timestamps = self.ed_flights()[["timestamp"]]
        expected_values = {
            "min": pd.Timestamp("2018-01-01 00:00:00"),
            "mean": pd.Timestamp("2018-01-21 19:20:45.564438232"),
            "max": pd.Timestamp("2018-02-11 23:50:12"),
            "nunique": 12236,
        }
        ed_metric = ed_timestamps.agg([agg])

        if agg == "nunique":
            assert ed_metric.dtypes["timestamp"] == np.int64
        else:
            assert ed_metric.dtypes["timestamp"] == np.dtype("datetime64[ns]")
        assert ed_metric["timestamp"][0] == expected_values[agg]

    @pytest.mark.parametrize("agg", ["mean", "min", "max"])
    def test_flights_datetime_metrics_agg_func(self, agg):
        ed_timestamps = self.ed_flights()[["timestamp"]]
        expected_values = {
            "min": pd.Timestamp("2018-01-01 00:00:00"),
            "mean": pd.Timestamp("2018-01-21 19:20:45.564438232"),
            "max": pd.Timestamp("2018-02-11 23:50:12"),
        }
        ed_metric = getattr(ed_timestamps, agg)(numeric_only=False)

        assert ed_metric.dtype == np.dtype("datetime64[ns]")
        assert ed_metric[0] == expected_values[agg]

    def test_flights_datetime_metrics_median(self):
        ed_df = self.ed_flights_small()[["timestamp"]]

        median = ed_df.median(numeric_only=False)[0]
        assert isinstance(median, pd.Timestamp)
        assert (
            pd.to_datetime("2018-01-01 10:00:00.000")
            <= median
            <= pd.to_datetime("2018-01-01 12:00:00.000")
        )

        median = ed_df.agg(["mean"])["timestamp"][0]
        assert isinstance(median, pd.Timestamp)
        assert (
            pd.to_datetime("2018-01-01 10:00:00.000")
            <= median
            <= pd.to_datetime("2018-01-01 12:00:00.000")
        )

    def test_metric_agg_keep_dtypes(self):
        # max, min, and median maintain their dtypes
        df = self.ed_flights_small()[["AvgTicketPrice", "Cancelled", "dayOfWeek"]]
        assert df.min().tolist() == [131.81910705566406, False, 0]
        assert df.max().tolist() == [989.9527587890625, True, 0]
        assert df.median().tolist() == [550.276123046875, False, 0]
        all_agg = df.agg(["min", "max", "median"])
        assert all_agg.dtypes.tolist() == [
            np.dtype("float64"),
            np.dtype("bool"),
            np.dtype("int64"),
        ]
        assert all_agg.to_dict() == {
            "AvgTicketPrice": {
                "max": 989.9527587890625,
                "median": 550.276123046875,
                "min": 131.81910705566406,
            },
            "Cancelled": {"max": True, "median": False, "min": False},
            "dayOfWeek": {"max": 0, "median": 0, "min": 0},
        }
