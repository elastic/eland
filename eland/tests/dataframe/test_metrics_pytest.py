# File called _pytest for PyCharm compatability

from eland.tests.common import TestData


from pandas.util.testing import assert_series_equal


class TestDataFrameMean(TestData):

    def test_to_mean(self):
        pd_flights = self.pd_flights()
        ed_flights = self.ed_flights()

        pd_mean = pd_flights.mean()
        ed_mean = ed_flights.mean()

        assert_series_equal(pd_mean, ed_mean)

    def test_to_sum(self):
        pd_flights = self.pd_flights()
        ed_flights = self.ed_flights()

        pd_sum = pd_flights.sum(numeric_only=True)
        ed_sum = ed_flights.sum(numeric_only=True)

        assert_series_equal(pd_sum, ed_sum)

    def test_to_min(self):
        pd_flights = self.pd_flights()
        ed_flights = self.ed_flights()

        pd_min = pd_flights.min(numeric_only=True)
        ed_min = ed_flights.min(numeric_only=True)

        assert_series_equal(pd_min, ed_min)

    def test_to_max(self):
        pd_flights = self.pd_flights()
        ed_flights = self.ed_flights()

        pd_max = pd_flights.max(numeric_only=True)
        ed_max = ed_flights.max(numeric_only=True)

        assert_series_equal(pd_max, ed_max)

