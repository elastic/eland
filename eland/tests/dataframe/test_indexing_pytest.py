# File called _pytest for PyCharm compatability
from eland.tests.common import TestData

import pandas as pd
import io

from pandas.util.testing import (
    assert_series_equal, assert_frame_equal)

class TestDataFrameIndexing(TestData):

    def test_mapping(self):
        ed_flights_mappings = pd.DataFrame(self.ed_flights()._mappings._mappings_capabilities
                                           [self.ed_flights()._mappings._mappings_capabilities._source==True]
                                           ['pd_dtype'])
        pd_flights_mappings = pd.DataFrame(self.pd_flights().dtypes, columns = ['pd_dtype'])

        assert_frame_equal(pd_flights_mappings, ed_flights_mappings)

        # We don't compare ecommerce here as the default dtypes in pandas from read_json
        # don't match the mapping types. This is mainly because the products field is
        # nested and so can be treated as a multi-field in ES, but not in pandas

    def test_head(self):
        pd_flights_head = self.pd_flights().head()
        ed_flights_head = self.ed_flights().head()

        print(ed_flights_head)

        assert_frame_equal(pd_flights_head, ed_flights_head)

        pd_ecommerce_head = self.pd_ecommerce().head()
        ed_ecommerce_head = self.ed_ecommerce().head()

        assert_frame_equal(pd_ecommerce_head, ed_ecommerce_head)

    def test_tail(self):
        pd_flights_tail = self.pd_flights().tail()
        ed_flights_tail = self.ed_flights().tail()

        print(ed_flights_tail)

        assert_frame_equal(pd_flights_tail, ed_flights_tail)

        pd_ecommerce_tail = self.pd_ecommerce().tail()
        ed_ecommerce_tail = self.ed_ecommerce().tail()

        assert_frame_equal(pd_ecommerce_tail, ed_ecommerce_tail)

    def test_describe(self):
        pd_flights_describe = self.pd_flights().describe()
        ed_flights_describe = self.ed_flights().describe()

        print(ed_flights_describe)

        # TODO - this fails now as ES aggregations are approximate
        #        if ES percentile agg uses
        #        "hdr": {
        #           "number_of_significant_value_digits": 3
        #         }
        #        this works
        #assert_almost_equal(pd_flights_describe, ed_flights_describe)

        pd_ecommerce_describe = self.pd_ecommerce().describe()
        ed_ecommerce_describe = self.ed_ecommerce().describe()

        print(ed_ecommerce_describe)

        # We don't compare ecommerce here as the default dtypes in pandas from read_json
        # don't match the mapping types. This is mainly because the products field is
        # nested and so can be treated as a multi-field in ES, but not in pandas

    def test_size(self):
        assert self.pd_flights().shape == self.ed_flights().shape
        assert len(self.pd_flights()) == len(self.ed_flights())

    def test_to_string(self):
        print(self.ed_flights())
        print(self.ed_flights().to_string())

    def test_info(self):
        ed_flights_info_buf = io.StringIO()
        pd_flights_info_buf = io.StringIO()

        self.ed_flights().info(buf=ed_flights_info_buf)
        self.pd_flights().info(buf=pd_flights_info_buf)

        print(ed_flights_info_buf.getvalue())

        ed_flights_info = (ed_flights_info_buf.getvalue().splitlines())
        pd_flights_info = (pd_flights_info_buf.getvalue().splitlines())

        flights_diff = set(ed_flights_info).symmetric_difference(set(pd_flights_info))

        ed_ecommerce_info_buf = io.StringIO()
        pd_ecommerce_info_buf = io.StringIO()

        self.ed_ecommerce().info(buf=ed_ecommerce_info_buf)
        self.pd_ecommerce().info(buf=pd_ecommerce_info_buf)

        ed_ecommerce_info = (ed_ecommerce_info_buf.getvalue().splitlines())
        pd_ecommerce_info = (pd_ecommerce_info_buf.getvalue().splitlines())

        # We don't compare ecommerce here as the default dtypes in pandas from read_json
        # don't match the mapping types. This is mainly because the products field is
        # nested and so can be treated as a multi-field in ES, but not in pandas
        ecommerce_diff = set(ed_ecommerce_info).symmetric_difference(set(pd_ecommerce_info))


    def test_count(self):
        pd_flights_count = self.pd_flights().count()
        ed_flights_count = self.ed_flights().count()

        assert_series_equal(pd_flights_count, ed_flights_count)

        pd_ecommerce_count = self.pd_ecommerce().count()
        ed_ecommerce_count = self.ed_ecommerce().count()

        assert_series_equal(pd_ecommerce_count, ed_ecommerce_count)

    def test_get_dtype_counts(self):
        pd_flights_get_dtype_counts = self.pd_flights().get_dtype_counts().sort_index()
        ed_flights_get_dtype_counts = self.ed_flights().get_dtype_counts().sort_index()

        assert_series_equal(pd_flights_get_dtype_counts, ed_flights_get_dtype_counts)

    def test_get_properties(self):
        pd_flights_shape = self.pd_flights().shape
        ed_flights_shape = self.ed_flights().shape

        assert pd_flights_shape == ed_flights_shape

        pd_flights_columns = self.pd_flights().columns
        ed_flights_columns = self.ed_flights().columns

        assert pd_flights_columns.tolist() == ed_flights_columns.tolist()

        pd_flights_dtypes = self.pd_flights().dtypes
        ed_flights_dtypes = self.ed_flights().dtypes

        assert_series_equal(pd_flights_dtypes, ed_flights_dtypes)

    def test_index(self):
        pd_flights = self.pd_flights()
        pd_flights_timestamp = pd_flights.set_index('timestamp')
        pd_flights.info()
        pd_flights_timestamp.info()
        pd_flights.info()

        ed_flights = self.ed_flights()
        ed_flights_timestamp = ed_flights.set_index('timestamp')
        ed_flights.info()
        ed_flights_timestamp.info()
        ed_flights.info()

