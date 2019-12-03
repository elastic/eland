# File called _pytest for PyCharm compatability
import eland as ed
from eland.tests import ES_TEST_CLIENT
from eland.tests import FLIGHTS_INDEX_NAME
from eland.tests.common import TestData
from eland.tests.common import assert_pandas_eland_series_equal


class TestSeriesRename(TestData):

    def test_rename(self):
        pd_carrier = self.pd_flights()['Carrier']
        ed_carrier = ed.Series(ES_TEST_CLIENT, FLIGHTS_INDEX_NAME, 'Carrier')

        assert_pandas_eland_series_equal(pd_carrier, ed_carrier)

        pd_renamed = pd_carrier.rename("renamed")
        ed_renamed = ed_carrier.rename("renamed")

        assert_pandas_eland_series_equal(pd_renamed, ed_renamed)
