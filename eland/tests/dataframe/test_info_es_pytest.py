# File called _pytest for PyCharm compatability

from eland.tests.common import TestData


class TestDataFrameInfoEs(TestData):

    def test_to_info1(self):
        ed_flights = self.ed_flights()

        head = ed_flights.head(103)
        slice = head[['timestamp', 'OriginRegion', 'Carrier']]
        iloc = slice.iloc[10:92, [0,2]]
        print(iloc.info_es())
        print(iloc)
