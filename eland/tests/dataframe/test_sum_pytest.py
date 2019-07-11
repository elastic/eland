# File called _pytest for PyCharm compatability

from eland.tests.common import TestData


from pandas.util.testing import assert_series_equal


class TestDataFrameSum(TestData):

    def test_to_mean1(self):
        pd_flights = self.pd_flights()
        ed_flights = self.ed_flights()

        pd_sum = pd_flights.sum(numeric_only=True)
        ed_sum = ed_flights.sum(numeric_only=True)

        assert_series_equal(pd_sum, ed_sum)


