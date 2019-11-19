# File called _pytest for PyCharm compatability
import eland as ed
from eland.tests.common import TestData


class TestSeriesValueCounts(TestData):

    def test_value_counts(self):
        pd_s = self.pd_flights()['Carrier']
        ed_s = self.ed_flights()['Carrier']

        pd_vc = pd_s.value_counts().to_string()
        ed_vc = ed_s.value_counts().to_string()

        print(type(pd_vc))
        print(type(ed_vc))

        assert pd_vc == ed_vc

    def test_value_counts_size(self):
        pd_s = self.pd_flights()['Carrier']
        ed_s = self.ed_flights()['Carrier']

        pd_vc = pd_s.value_counts()[:1].to_string()
        ed_vc = ed_s.value_counts(size=1).to_string()

        print(type(pd_vc))
        print(type(ed_vc))

        assert pd_vc == ed_vc