# File called _pytest for PyCharm compatability

from pandas.util.testing import assert_series_equal

from eland.tests.common import TestData

import eland as ed


class TestDataFrameMetrics(TestData):

    def test_mean(self):
        pd_flights = self.pd_flights()
        ed_flights = self.ed_flights()

        pd_mean = pd_flights.mean(numeric_only=True)
        ed_mean = ed_flights.mean(numeric_only=True)

        assert_series_equal(pd_mean, ed_mean)

    def test_sum(self):
        pd_flights = self.pd_flights()
        ed_flights = self.ed_flights()

        pd_sum = pd_flights.sum(numeric_only=True)
        ed_sum = ed_flights.sum(numeric_only=True)

        assert_series_equal(pd_sum, ed_sum)

    def test_min(self):
        pd_flights = self.pd_flights()
        ed_flights = self.ed_flights()

        pd_min = pd_flights.min(numeric_only=True)
        ed_min = ed_flights.min(numeric_only=True)

        assert_series_equal(pd_min, ed_min)

    def test_max(self):
        pd_flights = self.pd_flights()
        ed_flights = self.ed_flights()

        pd_max = pd_flights.max(numeric_only=True)
        ed_max = ed_flights.max(numeric_only=True)

        assert_series_equal(pd_max, ed_max)

    def test_ecommerce_selected_non_numeric_source_fields_max(self):
        # None of these are numeric
        columns = ['category', 'currency', 'customer_birth_date', 'customer_first_name', 'user']

        pd_ecommerce = self.pd_ecommerce()[columns]
        ed_ecommerce = self.ed_ecommerce()[columns]

        assert_series_equal(pd_ecommerce.max(numeric_only=True), ed_ecommerce.max(numeric_only=True))

    def test_ecommerce_selected_mixed_numeric_source_fields_max(self):
        # Some of these are numeric
        columns = ['category', 'currency', 'taxless_total_price', 'customer_birth_date',
                   'total_quantity', 'customer_first_name', 'user']

        pd_ecommerce = self.pd_ecommerce()[columns]
        ed_ecommerce = self.ed_ecommerce()[columns]

        assert_series_equal(pd_ecommerce.max(numeric_only=True), ed_ecommerce.max(numeric_only=True),
                            check_less_precise=True)


    def test_ecommerce_selected_all_numeric_source_fields_max(self):
        # All of these are numeric
        columns = ['total_quantity', 'taxful_total_price', 'taxless_total_price']

        pd_ecommerce = self.pd_ecommerce()[columns]
        ed_ecommerce = self.ed_ecommerce()[columns]

        assert_series_equal(pd_ecommerce.max(numeric_only=True), ed_ecommerce.max(numeric_only=True),
                            check_less_precise=True)
