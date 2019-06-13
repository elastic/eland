import pytest

import eland as ed

from eland.tests.dataframe.common import TestData

from pandas.util.testing import (
    assert_almost_equal, assert_frame_equal, assert_series_equal)

class TestDataFrameIndexing(TestData):

    def test_head(self):
        pd_head = self.pd_df.head()
        ed_head = self.ed_df.head()

        assert_frame_equal(pd_head, ed_head)

        

