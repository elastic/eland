# File called _pytest for PyCharm compatability

from pandas.util.testing import assert_series_equal

from eland.tests.common import TestData


class TestDataFrameDtypes(TestData):

    def test_flights_dtypes(self):
        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()

        assert_series_equal(pd_flights.dtypes, ed_flights.dtypes)
