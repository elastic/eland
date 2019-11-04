# File called _pytest for PyCharm compatability

from eland.tests.common import TestData


class TestDataFrameGet(TestData):

    def test_get_one_attribute(self):
        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()

        ed_get0 = ed_flights.get('Carrier')
        pd_get0 = pd_flights.get('Carrier')

        print(ed_get0, type(ed_get0))
        print(pd_get0, type(pd_get0))
