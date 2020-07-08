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

import numpy as np

from eland.tests.common import TestData


class TestSeriesMetrics(TestData):
    all_funcs = ["max", "min", "mean", "sum", "nunique", "var", "std", "mad"]
    timestamp_funcs = ["max", "min", "mean", "nunique"]

    def assert_almost_equal_for_agg(self, func, pd_metric, ed_metric):
        if func in ("nunique", "var", "mad"):
            np.testing.assert_almost_equal(pd_metric, ed_metric, decimal=-3)
        else:
            np.testing.assert_almost_equal(pd_metric, ed_metric, decimal=2)

    def test_flights_metrics(self):
        pd_flights = self.pd_flights()["AvgTicketPrice"]
        ed_flights = self.ed_flights()["AvgTicketPrice"]

        for func in self.all_funcs:
            pd_metric = getattr(pd_flights, func)()
            ed_metric = getattr(ed_flights, func)()

            self.assert_almost_equal_for_agg(func, pd_metric, ed_metric)

    def test_flights_timestamp(self):
        pd_flights = self.pd_flights()["timestamp"]
        ed_flights = self.ed_flights()["timestamp"]

        for func in self.timestamp_funcs:
            pd_metric = getattr(pd_flights, func)()
            ed_metric = getattr(ed_flights, func)()

            if hasattr(pd_metric, "floor"):
                pd_metric = pd_metric.floor("S")  # floor or pandas mean with have ns

            if func == "nunique":
                self.assert_almost_equal_for_agg(func, pd_metric, ed_metric)
            else:
                assert pd_metric == ed_metric

    def test_ecommerce_selected_non_numeric_source_fields(self):
        # None of these are numeric, will result in NaNs
        column = "category"

        ed_ecommerce = self.ed_ecommerce()[column]

        for func in self.all_funcs:
            if func == "nunique":  # nunique never returns 'NaN'
                continue

            ed_metric = getattr(ed_ecommerce, func)()
            print(func, ed_metric)
            assert np.isnan(ed_metric)

    def test_ecommerce_selected_all_numeric_source_fields(self):
        # All of these are numeric
        columns = ["total_quantity", "taxful_total_price", "taxless_total_price"]

        for column in columns:
            pd_ecommerce = self.pd_ecommerce()[column]
            ed_ecommerce = self.ed_ecommerce()[column]

            for func in self.all_funcs:
                pd_metric = getattr(pd_ecommerce, func)()
                ed_metric = getattr(ed_ecommerce, func)()
                self.assert_almost_equal_for_agg(func, pd_metric, ed_metric)
