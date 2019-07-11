# File called _pytest for PyCharm compatability
import pandas as pd
import eland as ed

from eland.tests.common import TestData
from eland.tests.common import assert_pandas_eland_frame_equal

from eland.tests import ELASTICSEARCH_HOST
from eland.tests import FLIGHTS_INDEX_NAME

from pandas.util.testing import assert_series_equal



class TestSeriesRepr(TestData):

    def test_repr(self):
        pd_s = self.pd_flights()['Carrier']
        ed_s = ed.Series(ELASTICSEARCH_HOST, FLIGHTS_INDEX_NAME, 'Carrier')

        pd_repr = repr(pd_s)
        ed_repr = repr(ed_s)

        assert pd_repr == ed_repr
