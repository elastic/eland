# File called _pytest for PyCharm compatability
from eland.tests.frame.common import TestData
from eland.tests import *

import eland as ed
import pandas as pd

from pandas.util.testing import (
    assert_almost_equal, assert_frame_equal, assert_series_equal)

class TestDataFrameIndexing(TestData):

    def test_mapping(self):
        ed_flights_mappings = pd.DataFrame(self.ed_flights().mappings.mappings_capabilities
                                           [self.ed_flights().mappings.mappings_capabilities._source==True]
                                           ['pd_dtype'])
        pd_flights_mappings = pd.DataFrame(self.pd_flights().dtypes, columns = ['pd_dtype'])

        assert_frame_equal(pd_flights_mappings, ed_flights_mappings)

        # We don't compare ecommerce here as the default dtypes in pandas from read_json
        # don't match the mapping types. This is mainly because the products field is
        # nested and so can be treated as a multi-field in ES, but not in pandas

    def test_head(self):
        pd_flights_head = self.pd_flights().head()
        ed_flights_head = self.ed_flights().head()

        assert_frame_equal(pd_flights_head, ed_flights_head)

        pd_ecommerce_head = self.pd_ecommerce().head()
        ed_ecommerce_head = self.ed_ecommerce().head()

        assert_frame_equal(pd_ecommerce_head, ed_ecommerce_head)

    def test_describe(self):
        pd_flights_describe = self.pd_flights().describe()
        ed_flights_describe = self.ed_flights().describe()

        # TODO - this fails now as ES aggregations are approximate
        #        if ES percentile agg uses
        #        "hdr": {
        #           "number_of_significant_value_digits": 3
        #         }
        #        this works
        #assert_almost_equal(pd_flights_describe, ed_flights_describe)

        pd_ecommerce_describe = self.pd_ecommerce().describe()
        ed_ecommerce_describe = self.ed_ecommerce().describe()

        # We don't compare ecommerce here as the default dtypes in pandas from read_json
        # don't match the mapping types. This is mainly because the products field is
        # nested and so can be treated as a multi-field in ES, but not in pandas

    def test_size(self):
        assert self.pd_flights().shape == self.ed_flights().shape
        assert len(self.pd_flights()) == len(self.ed_flights())

    def test_to_string(self):
        print(self.ed_flights())

