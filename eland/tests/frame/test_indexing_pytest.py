# File called _pytest for PyCharm compatability
from eland.tests.frame.common import TestData

from eland.tests import *

import eland as ed
import pandas as pd

from pandas.util.testing import (
    assert_almost_equal, assert_frame_equal, assert_series_equal)

class TestDataFrameIndexing(TestData):

    def test_head(self):
        pd_flights_head = self.pd_flights().head()
        ed_flights_head = self.ed_flights().head()

        assert_frame_equal(pd_flights_head, ed_flights_head)

        pd_ecommerce_head = self.pd_ecommerce().head()
        ed_ecommerce_head = self.ed_ecommerce().head()

        assert_frame_equal(pd_ecommerce_head, ed_ecommerce_head)

    def test_mappings(self):
        test_mapping1 = ed.read_es(ELASTICSEARCH_HOST, TEST_MAPPING1_INDEX_NAME)
        assert_frame_equal(TEST_MAPPING1_EXPECTED_DF, test_mapping1.fields)

