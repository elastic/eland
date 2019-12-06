# File called _pytest for PyCharm compatability
import pytest
import numpy as np

from eland.tests.common import TestData, assert_pandas_eland_series_equal


class TestSeriesArithmetics(TestData):

    def test_invalid_add_num(self):
        with pytest.raises(TypeError):
            assert 2 + self.ed_ecommerce()['currency']

        with pytest.raises(TypeError):
            assert self.ed_ecommerce()['currency'] + 2

        with pytest.raises(TypeError):
            assert self.ed_ecommerce()['currency'] + self.ed_ecommerce()['total_quantity']

        with pytest.raises(TypeError):
            assert self.ed_ecommerce()['total_quantity'] + self.ed_ecommerce()['currency']

    def test_str_add_ser(self):

        edadd = self.ed_ecommerce()['customer_first_name'] + self.ed_ecommerce()['customer_last_name']
        pdadd = self.pd_ecommerce()['customer_first_name'] + self.pd_ecommerce()['customer_last_name']

        assert_pandas_eland_series_equal(pdadd, edadd)

    def test_ser_add_str(self):
        edadd = self.ed_ecommerce()['customer_first_name'] + " is the first name."
        pdadd = self.pd_ecommerce()['customer_first_name'] + " is the first name."

        assert_pandas_eland_series_equal(pdadd, edadd)

    def test_ser_add_ser(self):
        edadd = "The last name is: " + self.ed_ecommerce()['customer_last_name']
        pdadd = "The last name is: " + self.pd_ecommerce()['customer_last_name']

        assert_pandas_eland_series_equal(pdadd, edadd)

    def test_non_aggregatable_add_str(self):
        with pytest.raises(ValueError):
            assert self.ed_ecommerce()['customer_gender'] + "is the gender"

    def teststr_add_non_aggregatable(self):
        with pytest.raises(ValueError):
            assert "The gender is: " + self.ed_ecommerce()['customer_gender']

    def test_non_aggregatable_add_aggregatable(self):
        with pytest.raises(ValueError):
            assert self.ed_ecommerce()['customer_gender'] + self.ed_ecommerce()['customer_first_name']

    def test_aggregatable_add_non_aggregatable(self):
            with pytest.raises(ValueError):
                assert self.ed_ecommerce()['customer_first_name'] + self.ed_ecommerce()['customer_gender']
