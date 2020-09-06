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
    filter_data = [
        "AvgTicketPrice",
        "Cancelled",
        "dayOfWeek",
        "timestamp",
        "DestCountry",
    ]

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
            ed_metric = getattr(ed_flights_1, func)(numeric_only=False)

            assert_series_equal(pd_metric, ed_metric, check_exact=False)

        # Test on zero rows to test NaN behaviour of sample std/variance
        pd_flights_0 = pd_flights[pd_flights.FlightNum == "XXX"][["AvgTicketPrice"]]
        ed_flights_0 = ed_flights[ed_flights.FlightNum == "XXX"][["AvgTicketPrice"]]

        for func in self.extended_funcs:
            pd_metric = getattr(pd_flights_0, func)()
            ed_metric = getattr(ed_flights_0, func)(numeric_only=False)

            assert_series_equal(pd_metric, ed_metric, check_exact=False)

    def test_ecommerce_selected_non_numeric_source_fields(self):
        # None of these are numeric
        columns = [
            "category",
            "currency",
            "customer_birth_date",
            "customer_first_name",
            "user",
        ]

        pd_ecommerce = self.pd_ecommerce().filter(columns)
        ed_ecommerce = self.ed_ecommerce().filter(columns)

        for func in self.funcs:
            assert_series_equal(
                getattr(pd_ecommerce, func)(numeric_only=True),
                getattr(ed_ecommerce, func)(numeric_only=True),
                check_exact=False,
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

        pd_ecommerce = self.pd_ecommerce().filter(columns)
        ed_ecommerce = self.ed_ecommerce().filter(columns)

        for func in self.funcs:
            assert_series_equal(
                getattr(pd_ecommerce, func)(numeric_only=True),
                getattr(ed_ecommerce, func)(numeric_only=True),
                check_exact=False,
            )

    def test_ecommerce_selected_all_numeric_source_fields(self):
        # All of these are numeric
        columns = ["total_quantity", "taxful_total_price", "taxless_total_price"]

        pd_ecommerce = self.pd_ecommerce().filter(columns)
        ed_ecommerce = self.ed_ecommerce().filter(columns)

        for func in self.funcs:
            assert_series_equal(
                getattr(pd_ecommerce, func)(numeric_only=True),
                getattr(ed_ecommerce, func)(numeric_only=True),
                check_exact=False,
            )

    def test_flights_datetime_metrics_agg(self):
        ed_timestamps = self.ed_flights()[["timestamp"]]
        expected_values = {
            "max": pd.Timestamp("2018-02-11 23:50:12"),
            "min": pd.Timestamp("2018-01-01 00:00:00"),
            "mean": pd.Timestamp("2018-01-21 19:20:45.564438232"),
            "sum": pd.NaT,
            "mad": pd.NaT,
            "var": pd.NaT,
            "std": pd.NaT,
            "nunique": 12236,
        }

        ed_metrics = ed_timestamps.agg(
            self.funcs + self.extended_funcs + ["nunique"], numeric_only=False
        )
        ed_metrics_dict = ed_metrics["timestamp"].to_dict()
        ed_metrics_dict.pop("median")  # Median is tested below.
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
        # max, min, and median maintain their dtypes for numeric_only=None
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

    def test_flights_numeric_only(self):
        filter_data = [
            "AvgTicketPrice",
            "Cancelled",
            "dayOfWeek",
            "timestamp",
            "DestCountry",
        ]
        # All Aggregations Data Check
        ed_flights = self.ed_flights().filter(filter_data)
        pd_flights = self.pd_flights().filter(filter_data)
        # agg => numeric_only True returns float64 values
        # We compare it with individual non-agg functions of pandas with numeric_only=True
        # not checking mad because it returns nan value for booleans.
        filtered_aggs = self.funcs + self.extended_funcs
        filtered_aggs.remove("mad")
        agg_data = ed_flights.agg(filtered_aggs, numeric_only=True).transpose()
        for agg in filtered_aggs:
            assert_series_equal(
                agg_data[agg].rename(None),
                getattr(pd_flights, agg)(
                    **({"numeric_only": True} if agg != "mad" else {})
                ),
                check_exact=False,
                rtol=True,
            )

    # Mean
    @pytest.mark.parametrize("numeric_only", [True, False, None])
    def test_mean_numeric_only(self, numeric_only):
        ed_flights = self.ed_flights().filter(self.filter_data)
        if numeric_only is True:
            calculated_values = ed_flights.mean(numeric_only=numeric_only)
            assert calculated_values.to_list() == [
                628.2536888148849,
                0.1284937590933456,
                2.835975189524466,
            ]
            dtype_list = [calculated_values[i].dtype for i in calculated_values.index]
            assert dtype_list == [
                np.dtype("float64"),
                np.dtype("float64"),
                np.dtype("float64"),
            ]
        elif numeric_only is False:
            calculated_values = ed_flights.mean(numeric_only=numeric_only)
            assert isinstance(calculated_values["timestamp"], pd.Timestamp)
            assert np.isnan(calculated_values["DestCountry"])
            calculated_values = calculated_values.drop(["timestamp", "DestCountry"])
            assert calculated_values.to_list() == [
                628.2536888148849,
                0.1284937590933456,
                2.835975189524466,
            ]
            assert isinstance(calculated_values["AvgTicketPrice"], float)
            assert isinstance(calculated_values["dayOfWeek"], float)
            assert isinstance(calculated_values["Cancelled"], float)
        elif numeric_only is None:
            calculated_values = ed_flights.mean(numeric_only=numeric_only)
            assert calculated_values.to_list() == [
                628.2536888148849,
                0.1284937590933456,
                2.835975189524466,
                pd.Timestamp("2018-01-21 19:20:45.564438232"),
            ]
            assert isinstance(calculated_values["timestamp"], pd.Timestamp)
            calculated_values = calculated_values.drop("timestamp")
            assert isinstance(calculated_values["AvgTicketPrice"], float)
            assert isinstance(calculated_values["dayOfWeek"], float)
            assert isinstance(calculated_values["Cancelled"], float)

    # Min
    @pytest.mark.parametrize("numeric_only", [True, False, None])
    def test_min_numeric_only(self, numeric_only):
        ed_flights = self.ed_flights().filter(self.filter_data)
        if numeric_only is True:
            calculated_values = ed_flights.min(numeric_only=numeric_only)
            assert calculated_values.to_list() == [100.0205307006836, 0.0, 0.0]
            dtype_list = [calculated_values[i].dtype for i in calculated_values.index]
            assert dtype_list == [
                np.dtype("float64"),
                np.dtype("float64"),
                np.dtype("float64"),
            ]
        elif numeric_only is False:
            calculated_values = ed_flights.min(numeric_only=numeric_only)
            assert isinstance(calculated_values["timestamp"], pd.Timestamp)
            assert np.isnan(calculated_values["DestCountry"])
            calculated_values = calculated_values.drop(["timestamp", "DestCountry"])
            assert calculated_values.to_list() == [100.0205307006836, 0, False]
            assert isinstance(calculated_values["AvgTicketPrice"], np.float64)
            assert isinstance(calculated_values["dayOfWeek"], np.int64)
            assert isinstance(calculated_values["Cancelled"], np.bool_)
        elif numeric_only is None:
            calculated_values = ed_flights.min(numeric_only=numeric_only)
            assert calculated_values.to_list() == [
                100.0205307006836,
                0,
                False,
                pd.Timestamp("2018-01-01 00:00:00"),
            ]
            assert isinstance(calculated_values["timestamp"], pd.Timestamp)
            assert isinstance(calculated_values["AvgTicketPrice"], np.float64)
            assert isinstance(calculated_values["dayOfWeek"], np.int64)
            assert isinstance(calculated_values["Cancelled"], np.bool_)

    # max
    @pytest.mark.parametrize("numeric_only", [True, False, None])
    def test_max_numeric_only(self, numeric_only):
        ed_flights = self.ed_flights().filter(self.filter_data)
        if numeric_only is True:
            calculated_values = ed_flights.max(numeric_only=numeric_only)
            assert calculated_values.to_list() == [1199.72900390625, 1.0, 6.0]
            dtype_list = [calculated_values[i].dtype for i in calculated_values.index]
            assert dtype_list == [
                np.dtype("float64"),
                np.dtype("float64"),
                np.dtype("float64"),
            ]
        elif numeric_only is False:
            calculated_values = ed_flights.max(numeric_only=numeric_only)
            assert isinstance(calculated_values["timestamp"], pd.Timestamp)
            assert np.isnan(calculated_values["DestCountry"])
            calculated_values = calculated_values.drop(["timestamp", "DestCountry"])
            assert calculated_values.to_list() == [1199.72900390625, True, 6]
            assert isinstance(calculated_values["AvgTicketPrice"], np.float64)
            assert isinstance(calculated_values["dayOfWeek"], np.int64)
            assert isinstance(calculated_values["Cancelled"], np.bool_)
        elif numeric_only is None:
            calculated_values = ed_flights.max(numeric_only=numeric_only)
            assert isinstance(calculated_values["timestamp"], pd.Timestamp)
            calculated_values = calculated_values.drop("timestamp")
            assert calculated_values.to_list() == [1199.72900390625, True, 6]
            assert isinstance(calculated_values["AvgTicketPrice"], np.float64)
            assert isinstance(calculated_values["dayOfWeek"], np.int64)
            assert isinstance(calculated_values["Cancelled"], np.bool_)

    # sum
    @pytest.mark.parametrize("numeric_only", [True, False, None])
    def test_sum_numeric_only(self, numeric_only):
        ed_flights = self.ed_flights().filter(self.filter_data)
        if numeric_only is True:
            calculated_values = ed_flights.sum(numeric_only=numeric_only)
            dtype_list = [calculated_values[i].dtype for i in calculated_values.index]
            assert calculated_values.to_list() == [8204364.922233582, 1678.0, 37035.0]
            assert dtype_list == [
                np.dtype("float64"),
                np.dtype("float64"),
                np.dtype("float64"),
            ]
        elif numeric_only is False:
            calculated_values = ed_flights.sum(numeric_only=numeric_only)
            assert pd.isnull(calculated_values["timestamp"])
            assert np.isnan(calculated_values["DestCountry"])
            calculated_values = calculated_values.drop(["timestamp", "DestCountry"])
            assert calculated_values.to_list() == [8204364.922233582, 1678, 37035]
            assert isinstance(calculated_values["AvgTicketPrice"], np.float64)
            assert isinstance(calculated_values["dayOfWeek"], np.int64)
            assert isinstance(calculated_values["Cancelled"], np.int64)
        elif numeric_only is None:
            calculated_values = ed_flights.sum(numeric_only=numeric_only)
            assert calculated_values.to_list() == [8204364.922233582, 1678, 37035]
            dtype_list = [calculated_values[i].dtype for i in calculated_values.index]
            assert dtype_list == [
                np.dtype("float64"),
                np.dtype("int64"),
                np.dtype("int64"),
            ]

    # std
    @pytest.mark.parametrize("numeric_only", [True, False, None])
    def test_std_numeric_only(self, numeric_only):
        ed_flights = self.ed_flights().filter(self.filter_data)
        if numeric_only is True:
            calculated_values = ed_flights.std(numeric_only=numeric_only)
            dtype_list = [calculated_values[i].dtype for i in calculated_values.index]
            assert calculated_values.to_list() == [
                266.4070611666801,
                0.33466440694020916,
                1.9395130445445228,
            ]
            assert dtype_list == [
                np.dtype("float64"),
                np.dtype("float64"),
                np.dtype("float64"),
            ]
        elif numeric_only is False:
            calculated_values = ed_flights.std(numeric_only=numeric_only)
            assert pd.isnull(calculated_values["timestamp"])
            assert np.isnan(calculated_values["DestCountry"])
            calculated_values = calculated_values.drop(["timestamp", "DestCountry"])
            assert calculated_values.to_list() == [
                266.4070611666801,
                0.33466440694020916,
                1.9395130445445228,
            ]
            assert isinstance(calculated_values["AvgTicketPrice"], float)
            assert isinstance(calculated_values["dayOfWeek"], float)
            assert isinstance(calculated_values["Cancelled"], float)
        elif numeric_only is None:
            calculated_values = ed_flights.std(numeric_only=numeric_only)
            assert calculated_values.to_list() == [
                266.4070611666801,
                0.33466440694020916,
                1.9395130445445228,
            ]
            assert isinstance(calculated_values["AvgTicketPrice"], float)
            assert isinstance(calculated_values["dayOfWeek"], float)
            assert isinstance(calculated_values["Cancelled"], float)

    # var
    @pytest.mark.parametrize("numeric_only", [True, False, None])
    def test_var_numeric_only(self, numeric_only):
        ed_flights = self.ed_flights().filter(self.filter_data)
        if numeric_only is True:
            calculated_values = ed_flights.var(numeric_only=numeric_only)
            dtype_list = [calculated_values[i].dtype for i in calculated_values.index]
            assert calculated_values.to_list() == [
                70964.57023354847,
                0.111987400797438,
                3.7612787756607213,
            ]
            assert dtype_list == [
                np.dtype("float64"),
                np.dtype("float64"),
                np.dtype("float64"),
            ]
        elif numeric_only is False:
            calculated_values = ed_flights.var(numeric_only=numeric_only)
            assert pd.isnull(calculated_values["timestamp"])
            assert np.isnan(calculated_values["DestCountry"])
            calculated_values = calculated_values.drop(["timestamp", "DestCountry"])
            assert calculated_values.to_list() == [
                70964.57023354847,
                0.111987400797438,
                3.7612787756607213,
            ]
            assert isinstance(calculated_values["AvgTicketPrice"], np.float64)
            assert isinstance(calculated_values["dayOfWeek"], np.float64)
            assert isinstance(calculated_values["Cancelled"], np.float64)
        elif numeric_only is None:
            calculated_values = ed_flights.var(numeric_only=numeric_only)
            assert calculated_values.to_list() == [
                70964.57023354847,
                0.111987400797438,
                3.7612787756607213,
            ]
            dtype_list = [calculated_values[i].dtype for i in calculated_values.index]
            assert dtype_list == [
                np.dtype("float64"),
                np.dtype("float64"),
                np.dtype("float64"),
            ]

    # median
    @pytest.mark.parametrize("numeric_only", [True, False, None])
    def test_median_numeric_only(self, numeric_only):
        ed_flights = self.ed_flights().filter(self.filter_data)
        if numeric_only is True:
            calculated_values = ed_flights.median(numeric_only=numeric_only)
            dtype_list = [calculated_values[i].dtype for i in calculated_values.index]
            assert (
                (calculated_values.to_dict()["AvgTicketPrice"] * 0.9)
                <= 640.3872852064159
                <= (calculated_values.to_dict()["AvgTicketPrice"] * 1.1)
            )
            assert calculated_values["Cancelled"] == 0.0
            assert calculated_values["dayOfWeek"] == 3.0
            assert dtype_list == [
                np.dtype("float64"),
                np.dtype("float64"),
                np.dtype("float64"),
            ]
        elif numeric_only is False:
            expected_values = {
                "AvgTicketPrice": 640.3222933002547,
                "Cancelled": False,
                "dayOfWeek": 3,
                "timestamp": pd.Timestamp("2018-01-21 23:58:10.414120850"),
            }
            calculated_values = ed_flights.median(numeric_only=numeric_only)
            assert isinstance(calculated_values["timestamp"], pd.Timestamp)
            assert np.isnan(calculated_values["DestCountry"])
            assert (
                expected_values["Cancelled"] == calculated_values.to_dict()["Cancelled"]
            )
            assert (
                (calculated_values.to_dict()["AvgTicketPrice"] * 0.9)
                <= expected_values["AvgTicketPrice"]
                <= (calculated_values.to_dict()["AvgTicketPrice"] * 1.1)
            )
            assert (
                pd.to_datetime("2018-01-21 23:00:00.000")
                <= expected_values["timestamp"]
                <= pd.to_datetime("2018-01-21 23:59:59.000")
            )
            assert (
                expected_values["dayOfWeek"] == calculated_values.to_dict()["dayOfWeek"]
            )
            assert isinstance(calculated_values["Cancelled"], np.bool_)
            assert isinstance(calculated_values["dayOfWeek"], np.int64)
        elif numeric_only is None:
            expected_values = {
                "AvgTicketPrice": 640.3872852064159,
                "Cancelled": False,
                "dayOfWeek": 3,
                "timestamp": pd.Timestamp("2018-01-21 23:58:10.414120850"),
            }
            calculated_values = ed_flights.median(numeric_only=numeric_only)
            assert isinstance(calculated_values["timestamp"], pd.Timestamp)
            assert (
                (calculated_values.to_dict()["AvgTicketPrice"] * 0.9)
                <= expected_values["AvgTicketPrice"]
                <= (calculated_values.to_dict()["AvgTicketPrice"] * 1.1)
            )
            assert (
                pd.to_datetime("2018-01-21 23:00:00.000")
                <= expected_values["timestamp"]
                <= pd.to_datetime("2018-01-21 23:59:00.000")
            )
            assert isinstance(calculated_values["Cancelled"], np.bool_)
            assert isinstance(calculated_values["dayOfWeek"], np.int64)

    # mad
    @pytest.mark.parametrize("numeric_only", [True, False, None])
    def test_mad_numeric_only(self, numeric_only):
        ed_flights = self.ed_flights().filter(self.filter_data)
        if numeric_only is True:
            expected_values = {"AvgTicketPrice": 213.47889841845912, "dayOfWeek": 2.0}
            calculated_values = ed_flights.mad(numeric_only=numeric_only)
            assert (
                expected_values["dayOfWeek"] == calculated_values.to_dict()["dayOfWeek"]
            )
            assert (
                (calculated_values["AvgTicketPrice"] * 0.9)
                <= expected_values["AvgTicketPrice"]
                <= (calculated_values["AvgTicketPrice"] * 1.1)
            )
            assert calculated_values["AvgTicketPrice"].dtype == np.dtype("float64")
        elif numeric_only is False:
            expected_values = {"AvgTicketPrice": 213.36870923117985, "dayOfWeek": 2.0}
            calculated_values = ed_flights.mad(numeric_only=numeric_only)
            assert pd.isnull(calculated_values["timestamp"])
            assert np.isnan(calculated_values["DestCountry"])
            assert np.isnan(calculated_values["Cancelled"])
            calculated_values = calculated_values.drop(
                ["timestamp", "DestCountry", "Cancelled"]
            )
            assert (
                expected_values["dayOfWeek"] == calculated_values.to_dict()["dayOfWeek"]
            )
            assert (
                (calculated_values["AvgTicketPrice"] * 0.9)
                <= expected_values["AvgTicketPrice"]
                <= (calculated_values["AvgTicketPrice"] * 1.1)
            )
            assert isinstance(calculated_values["AvgTicketPrice"], float)
            assert isinstance(calculated_values["dayOfWeek"], float)

        elif numeric_only is None:
            expected_values = {"AvgTicketPrice": 213.4408885767035, "dayOfWeek": 2.0}
            calculated_values = ed_flights.mad(numeric_only=numeric_only)
            assert (
                (calculated_values["AvgTicketPrice"] * 0.9)
                <= expected_values["AvgTicketPrice"]
                <= (calculated_values["AvgTicketPrice"] * 1.1)
            )
            assert isinstance(calculated_values["AvgTicketPrice"], float)
            assert isinstance(calculated_values["dayOfWeek"], float)
