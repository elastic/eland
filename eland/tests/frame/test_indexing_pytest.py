# File called _pytest for PyCharm compatability
from eland.tests.frame.common import TestData
from eland.tests import *

import eland as ed
import pandas as pd

from pandas.util.testing import (
    assert_almost_equal, assert_frame_equal, assert_series_equal)

class TestDataFrameIndexing(TestData):

    def test_results(self):
        test = ed.read_es(ELASTICSEARCH_HOST, TEST_NESTED_USER_GROUP_INDEX_NAME)

        print(test.mappings.mappings_capabilities)

        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)

        print(test.head())


    def test_head(self):
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)

        """
        pd_flights_head = self.pd_flights().head()
        ed_flights_head = self.ed_flights().head()

        assert_frame_equal(pd_flights_head, ed_flights_head)
        """

        pd_ecommerce_head = self.pd_ecommerce().head()
        ed_ecommerce_head = self.ed_ecommerce().head()

        print(self.ed_ecommerce().mappings.source_fields_pd_dtypes())

        print(ed_ecommerce_head.dtypes)
        print(pd_ecommerce_head.dtypes)

        #print(ed_ecommerce_head)

        assert_frame_equal(pd_ecommerce_head, ed_ecommerce_head)

    def test_mappings(self):
        test_mapping1 = ed.read_es(ELASTICSEARCH_HOST, TEST_MAPPING1_INDEX_NAME)
        assert_frame_equal(TEST_MAPPING1_EXPECTED_DF, test_mapping1.fields)
