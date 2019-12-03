# File called _pytest for PyCharm compatability
import eland as ed
from eland.tests import ES_TEST_CLIENT
from eland.tests import FLIGHTS_INDEX_NAME
from eland.tests.common import TestData
from eland.tests.common import assert_pandas_eland_series_equal


class TestSeriesName(TestData):

    def test_name(self):
        # deep copy pandas DataFrame as .name alters this reference frame
        pd_series = self.pd_flights()['Carrier'].copy(deep=True)
        ed_series = ed.Series(ES_TEST_CLIENT, FLIGHTS_INDEX_NAME, 'Carrier')

        assert_pandas_eland_series_equal(pd_series, ed_series)
        assert ed_series.name == pd_series.name

        pd_series.name = "renamed1"
        ed_series.name = "renamed1"

        assert_pandas_eland_series_equal(pd_series, ed_series)
        assert ed_series.name == pd_series.name

        pd_series.name = "renamed2"
        ed_series.name = "renamed2"

        assert_pandas_eland_series_equal(pd_series, ed_series)
        assert ed_series.name == pd_series.name
