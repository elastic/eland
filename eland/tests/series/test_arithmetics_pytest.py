# File called _pytest for PyCharm compatability
import eland as ed
from eland.tests.common import TestData, assert_pandas_eland_series_equal
from pandas.util.testing import assert_series_equal
import pytest

import numpy as np


class TestSeriesArithmetics(TestData):

    def test_ecommerce_series_invalid_div(self):
        pd_df = self.pd_ecommerce()
        ed_df = self.ed_ecommerce()

        # eland / pandas == error
        with pytest.raises(TypeError):
            ed_df['total_quantity'] / pd_df['taxful_total_price']

    def test_ecommerce_series_basic_arithmetics(self):
        pd_df = self.pd_ecommerce().head(100)
        ed_df = self.ed_ecommerce().head(100)

        ops = ['__add__',
               '__truediv__',
               '__floordiv__',
               '__pow__',
               '__mod__',
               '__mul__',
               '__sub__',
               'add',
               'truediv',
               'floordiv',
               'pow',
               'mod',
               'mul',
               'sub']

        for op in ops:
            pd_series = getattr(pd_df['taxful_total_price'], op)(pd_df['total_quantity'])
            ed_series = getattr(ed_df['taxful_total_price'], op)(ed_df['total_quantity'])
            assert_pandas_eland_series_equal(pd_series, ed_series, check_less_precise=True)

            pd_series = getattr(pd_df['taxful_total_price'], op)(10.56)
            ed_series = getattr(ed_df['taxful_total_price'], op)(10.56)
            assert_pandas_eland_series_equal(pd_series, ed_series, check_less_precise=True)

            pd_series = getattr(pd_df['taxful_total_price'], op)(int(8))
            ed_series = getattr(ed_df['taxful_total_price'], op)(int(8))
            assert_pandas_eland_series_equal(pd_series, ed_series, check_less_precise=True)

