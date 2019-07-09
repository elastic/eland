# File called _pytest for PyCharm compatability

from eland.tests.common import TestData


class TestDataFrameCount(TestData):

    def test_to_string1(self):
        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()

        #ed_count = ed_flights.count()

