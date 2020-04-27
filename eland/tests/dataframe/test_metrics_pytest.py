# Licensed to Elasticsearch B.V under one or more agreements.
# Elasticsearch B.V licenses this file to you under the Apache 2.0 License.
# See the LICENSE file in the project root for more information

# File called _pytest for PyCharm compatibility

from pandas.testing import assert_series_equal

from eland.tests.common import TestData


class TestDataFrameMetrics(TestData):
    funcs = ["max", "min", "mean", "sum"]
    extended_funcs = ["median", "mad", "var", "std"]

    def test_flights_metrics(self):
        pd_flights = self.pd_flights()
        ed_flights = self.ed_flights()

        for func in self.funcs:
            pd_metric = getattr(pd_flights, func)(numeric_only=True)
            ed_metric = getattr(ed_flights, func)(numeric_only=True)

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
