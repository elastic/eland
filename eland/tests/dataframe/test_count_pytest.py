# File called _pytest for PyCharm compatability

from pandas.util.testing import assert_series_equal

from eland.tests.common import TestData

import pandas as pd

class TestDataFrameCount(TestData):

    def test_ecommerce_count(self):
        pd_ecommerce = self.pd_ecommerce()
        ed_ecommerce = self.ed_ecommerce()

        pd_count = pd_ecommerce.count()
        ed_count = ed_ecommerce.count()

        assert_series_equal(pd_count, ed_count)
