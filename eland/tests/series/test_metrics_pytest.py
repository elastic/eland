# Licensed to Elasticsearch B.V under one or more agreements.
# Elasticsearch B.V licenses this file to you under the Apache 2.0 License.
# See the LICENSE file in the project root for more information

# File called _pytest for PyCharm compatability

import numpy as np

from eland.tests.common import TestData


class TestSeriesMetrics(TestData):
    funcs = ["max", "min", "mean", "sum"]
    timestamp_funcs = ["max", "min", "mean"]

    def test_flights_metrics(self):
        pd_flights = self.pd_flights()["AvgTicketPrice"]
        ed_flights = self.ed_flights()["AvgTicketPrice"]

        for func in self.funcs:
            pd_metric = getattr(pd_flights, func)()
            ed_metric = getattr(ed_flights, func)()
            np.testing.assert_almost_equal(pd_metric, ed_metric, decimal=2)

    def test_flights_timestamp(self):
        pd_flights = self.pd_flights()["timestamp"]
        ed_flights = self.ed_flights()["timestamp"]

        for func in self.timestamp_funcs:
            pd_metric = getattr(pd_flights, func)()
            ed_metric = getattr(ed_flights, func)()
            pd_metric = pd_metric.floor("S")  # floor or pandas mean with have ns
            assert pd_metric == ed_metric

    def test_ecommerce_selected_non_numeric_source_fields(self):
        # None of these are numeric
        column = "category"

        ed_ecommerce = self.ed_ecommerce()[column]

        for func in self.funcs:
            ed_metric = getattr(ed_ecommerce, func)()
            assert ed_metric.empty

    def test_ecommerce_selected_all_numeric_source_fields(self):
        # All of these are numeric
        columns = ["total_quantity", "taxful_total_price", "taxless_total_price"]

        for column in columns:
            pd_ecommerce = self.pd_ecommerce()[column]
            ed_ecommerce = self.ed_ecommerce()[column]

            for func in self.funcs:
                np.testing.assert_almost_equal(
                    getattr(pd_ecommerce, func)(),
                    getattr(ed_ecommerce, func)(),
                    decimal=2,
                )
