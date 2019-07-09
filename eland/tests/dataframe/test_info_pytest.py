# File called _pytest for PyCharm compatability

from eland.tests.common import TestData


class TestDataFrameInfo(TestData):

    def test_to_info1(self):
        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()

        ed_flights.info()
