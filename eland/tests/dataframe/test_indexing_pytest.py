# File called _pytest for PyCharm compatability
from eland.tests.dataframe.common import TestData

from pandas.util.testing import (
    assert_almost_equal, assert_frame_equal, assert_series_equal)

class TestDataFrameIndexing(TestData):

    def test_head(self):
        pd_flights_head = self.pd_flights().head()
        ed_flights_head = self.ed_flights().head()

        assert_frame_equal(pd_flights_head, ed_flights_head)

        pd_ecommerce_head = self.pd_ecommerce().head()
        ed_ecommerce_head = self.ed_ecommerce().head()

        #print(pd_ecommerce_head.dtypes)
        #print(ed_ecommerce_head.dtypes)

        assert_frame_equal(pd_ecommerce_head, ed_ecommerce_head)

        

