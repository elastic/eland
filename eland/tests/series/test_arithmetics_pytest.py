# File called _pytest for PyCharm compatability
import eland as ed
from eland.tests.common import TestData, assert_pandas_eland_series_equal
from pandas.util.testing import assert_series_equal
import pytest


class TestSeriesArithmetics(TestData):

    def test_ecommerce_series_invalid_div(self):
        pd_df = self.pd_ecommerce()
        ed_df = self.ed_ecommerce()

        # eland / pandas == error
        with pytest.raises(TypeError):
            ed_df['total_quantity'] / pd_df['taxful_total_price']

    def test_ecommerce_series_div(self):
        pd_df = self.pd_ecommerce()
        ed_df = self.ed_ecommerce()

        pd_avg_price = pd_df['total_quantity'] / pd_df['taxful_total_price']
        print(pd_avg_price) # this has None as name

        ed_avg_price = ed_df['total_quantity'] / ed_df['taxful_total_price']
        print(ed_avg_price)

        assert_pandas_eland_series_equal(pd_avg_price, ed_avg_price, check_less_precise=True)

    def test_ecommerce_series_div_float(self):
        pd_df = self.pd_ecommerce()
        ed_df = self.ed_ecommerce()

        pd_avg_price = pd_df['total_quantity'] / 10.0
        print(pd_avg_price)

        ed_avg_price = ed_df['total_quantity'] / 10.0
        print(ed_avg_price)

    def test_ecommerce_series_div_other(self):
        ed_df = self.ed_ecommerce()

        ed_s1 = ed_df.total_quantity
        ed_s2 = ed_df.taxful_total_price

        print(ed_s1)
        print(ed_s2)

        print(ed_s1)
        print(ed_s2)
