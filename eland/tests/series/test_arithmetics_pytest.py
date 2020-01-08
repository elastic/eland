#  Copyright 2019 Elasticsearch BV
#
#      Licensed under the Apache License, Version 2.0 (the "License");
#      you may not use this file except in compliance with the License.
#      You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#      Unless required by applicable law or agreed to in writing, software
#      distributed under the License is distributed on an "AS IS" BASIS,
#      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#      See the License for the specific language governing permissions and
#      limitations under the License.

# File called _pytest for PyCharm compatability
import numpy as np
import pytest

from eland.tests.common import TestData, assert_pandas_eland_series_equal


class TestSeriesArithmetics(TestData):

    def test_ecommerce_series_invalid_div(self):
        pd_df = self.pd_ecommerce()
        ed_df = self.ed_ecommerce()

        # eland / pandas == error
        with pytest.raises(TypeError):
            ed_series = ed_df['total_quantity'] / pd_df['taxful_total_price']

    def test_ecommerce_series_simple_arithmetics(self):
        pd_df = self.pd_ecommerce().head(100)
        ed_df = self.ed_ecommerce().head(100)

        pd_series = pd_df['taxful_total_price'] + 5 + pd_df['total_quantity'] / pd_df['taxless_total_price'] - pd_df[
            'total_unique_products'] * 10.0 + pd_df['total_quantity']
        ed_series = ed_df['taxful_total_price'] + 5 + ed_df['total_quantity'] / ed_df['taxless_total_price'] - ed_df[
            'total_unique_products'] * 10.0 + ed_df['total_quantity']

        assert_pandas_eland_series_equal(pd_series, ed_series, check_less_precise=True)

    def test_ecommerce_series_simple_integer_addition(self):
        pd_df = self.pd_ecommerce().head(100)
        ed_df = self.ed_ecommerce().head(100)

        pd_series = pd_df['taxful_total_price'] + 5
        ed_series = ed_df['taxful_total_price'] + 5

        assert_pandas_eland_series_equal(pd_series, ed_series, check_less_precise=True)

    def test_ecommerce_series_simple_series_addition(self):
        pd_df = self.pd_ecommerce().head(100)
        ed_df = self.ed_ecommerce().head(100)

        pd_series = pd_df['taxful_total_price'] + pd_df['total_quantity']
        ed_series = ed_df['taxful_total_price'] + ed_df['total_quantity']

        assert_pandas_eland_series_equal(pd_series, ed_series, check_less_precise=True)

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

            pd_series = getattr(pd_df['taxful_total_price'], op)(np.float32(1.879))
            ed_series = getattr(ed_df['taxful_total_price'], op)(np.float32(1.879))
            assert_pandas_eland_series_equal(pd_series, ed_series, check_less_precise=True)

            pd_series = getattr(pd_df['taxful_total_price'], op)(int(8))
            ed_series = getattr(ed_df['taxful_total_price'], op)(int(8))
            assert_pandas_eland_series_equal(pd_series, ed_series, check_less_precise=True)

    def test_supported_series_dtypes_ops(self):
        pd_df = self.pd_ecommerce().head(100)
        ed_df = self.ed_ecommerce().head(100)

        # Test some specific operations that are and aren't supported
        numeric_ops = ['__add__',
                       '__truediv__',
                       '__floordiv__',
                       '__pow__',
                       '__mod__',
                       '__mul__',
                       '__sub__']

        non_string_numeric_ops = ['__add__',
                                  '__truediv__',
                                  '__floordiv__',
                                  '__pow__',
                                  '__mod__',
                                  '__sub__']
        # __mul__ is supported for int * str in pandas

        # float op float
        for op in numeric_ops:
            pd_series = getattr(pd_df['taxful_total_price'], op)(pd_df['taxless_total_price'])
            ed_series = getattr(ed_df['taxful_total_price'], op)(ed_df['taxless_total_price'])
            assert_pandas_eland_series_equal(pd_series, ed_series, check_less_precise=True)

        # int op float
        for op in numeric_ops:
            pd_series = getattr(pd_df['total_quantity'], op)(pd_df['taxless_total_price'])
            ed_series = getattr(ed_df['total_quantity'], op)(ed_df['taxless_total_price'])
            assert_pandas_eland_series_equal(pd_series, ed_series, check_less_precise=True)

        # float op int
        for op in numeric_ops:
            pd_series = getattr(pd_df['taxful_total_price'], op)(pd_df['total_quantity'])
            ed_series = getattr(ed_df['taxful_total_price'], op)(ed_df['total_quantity'])
            assert_pandas_eland_series_equal(pd_series, ed_series, check_less_precise=True)

        # str op int (throws)
        for op in non_string_numeric_ops:
            with pytest.raises(TypeError):
                pd_series = getattr(pd_df['currency'], op)(pd_df['total_quantity'])
            with pytest.raises(TypeError):
                ed_series = getattr(ed_df['currency'], op)(ed_df['total_quantity'])
            with pytest.raises(TypeError):
                pd_series = getattr(pd_df['currency'], op)(1)
            with pytest.raises(TypeError):
                ed_series = getattr(ed_df['currency'], op)(1)

        # int op str (throws)
        for op in non_string_numeric_ops:
            with pytest.raises(TypeError):
                pd_series = getattr(pd_df['total_quantity'], op)(pd_df['currency'])
            with pytest.raises(TypeError):
                ed_series = getattr(ed_df['total_quantity'], op)(ed_df['currency'])

    def test_ecommerce_series_basic_rarithmetics(self):
        pd_df = self.pd_ecommerce().head(10)
        ed_df = self.ed_ecommerce().head(10)

        ops = ['__radd__',
               '__rtruediv__',
               '__rfloordiv__',
               '__rpow__',
               '__rmod__',
               '__rmul__',
               '__rsub__',
               'radd',
               'rtruediv',
               'rfloordiv',
               'rpow',
               'rmod',
               'rmul',
               'rsub']

        for op in ops:
            pd_series = getattr(pd_df['taxful_total_price'], op)(pd_df['total_quantity'])
            ed_series = getattr(ed_df['taxful_total_price'], op)(ed_df['total_quantity'])
            assert_pandas_eland_series_equal(pd_series, ed_series, check_less_precise=True)

            pd_series = getattr(pd_df['taxful_total_price'], op)(3.141)
            ed_series = getattr(ed_df['taxful_total_price'], op)(3.141)
            assert_pandas_eland_series_equal(pd_series, ed_series, check_less_precise=True)

            pd_series = getattr(pd_df['taxful_total_price'], op)(np.float32(2.879))
            ed_series = getattr(ed_df['taxful_total_price'], op)(np.float32(2.879))
            assert_pandas_eland_series_equal(pd_series, ed_series, check_less_precise=True)

            pd_series = getattr(pd_df['taxful_total_price'], op)(int(6))
            ed_series = getattr(ed_df['taxful_total_price'], op)(int(6))
            assert_pandas_eland_series_equal(pd_series, ed_series, check_less_precise=True)

    def test_supported_series_dtypes_rops(self):
        pd_df = self.pd_ecommerce().head(100)
        ed_df = self.ed_ecommerce().head(100)

        # Test some specific operations that are and aren't supported
        numeric_ops = ['__radd__',
                       '__rtruediv__',
                       '__rfloordiv__',
                       '__rpow__',
                       '__rmod__',
                       '__rmul__',
                       '__rsub__']

        non_string_numeric_ops = ['__radd__',
                                  '__rtruediv__',
                                  '__rfloordiv__',
                                  '__rpow__',
                                  '__rmod__',
                                  '__rsub__']
        # __rmul__ is supported for int * str in pandas

        # float op float
        for op in numeric_ops:
            pd_series = getattr(pd_df['taxful_total_price'], op)(pd_df['taxless_total_price'])
            ed_series = getattr(ed_df['taxful_total_price'], op)(ed_df['taxless_total_price'])
            assert_pandas_eland_series_equal(pd_series, ed_series, check_less_precise=True)

        # int op float
        for op in numeric_ops:
            pd_series = getattr(pd_df['total_quantity'], op)(pd_df['taxless_total_price'])
            ed_series = getattr(ed_df['total_quantity'], op)(ed_df['taxless_total_price'])
            assert_pandas_eland_series_equal(pd_series, ed_series, check_less_precise=True)

        # float op int
        for op in numeric_ops:
            pd_series = getattr(pd_df['taxful_total_price'], op)(pd_df['total_quantity'])
            ed_series = getattr(ed_df['taxful_total_price'], op)(ed_df['total_quantity'])
            assert_pandas_eland_series_equal(pd_series, ed_series, check_less_precise=True)

        # str op int (throws)
        for op in non_string_numeric_ops:
            with pytest.raises(TypeError):
                pd_series = getattr(pd_df['currency'], op)(pd_df['total_quantity'])
            with pytest.raises(TypeError):
                ed_series = getattr(ed_df['currency'], op)(ed_df['total_quantity'])
            with pytest.raises(TypeError):
                pd_series = getattr(pd_df['currency'], op)(10.0)
            with pytest.raises(TypeError):
                ed_series = getattr(ed_df['currency'], op)(10.0)

        # int op str (throws)
        for op in non_string_numeric_ops:
            with pytest.raises(TypeError):
                pd_series = getattr(pd_df['total_quantity'], op)(pd_df['currency'])
            with pytest.raises(TypeError):
                ed_series = getattr(ed_df['total_quantity'], op)(ed_df['currency'])
