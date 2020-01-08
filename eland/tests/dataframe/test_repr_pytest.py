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

import pandas as pd
import pytest

from eland.compat import PY36
from eland.dataframe import DEFAULT_NUM_ROWS_DISPLAYED
from eland.tests.common import TestData, assert_pandas_eland_series_equal


class TestDataFrameRepr(TestData):

    @classmethod
    def setup_class(cls):
        # conftest.py changes this default - restore to original setting
        pd.set_option('display.max_rows', 60)

    # Override these methods as:
    # {'lat': '-33.94609833', 'lon': '151.177002'} order is not consistent in python 3.5 (dict's not ordered)
    # remove from test for now
    def ed_flights(self):
        if not PY36:
            return super().ed_flights().drop(columns=['OriginLocation', 'DestLocation'])
        return super().ed_flights()

    def pd_flights(self):
        if not PY36:
            return super().pd_flights().drop(columns=['OriginLocation', 'DestLocation'])
        return super().pd_flights()

    """
    to_string
    """

    def test_simple_lat_lon(self):
        """
        Note on nested object order - this can change when
        note this could be a bug in ES...
        PUT my_index/doc/1
        {
          "location": {
            "lat": "50.033333",
            "lon": "8.570556"
          }
        }

        GET my_index/_search

        "_source": {
          "location": {
            "lat": "50.033333",
            "lon": "8.570556"
          }
        }

        GET my_index/_search
        {
          "_source": "location"
        }

        "_source": {
          "location": {
            "lon": "8.570556",
            "lat": "50.033333"
          }
        }

        Hence we store the pandas df source json as 'lon', 'lat'
        """
        if PY36:
            pd_dest_location = self.pd_flights()['DestLocation'].head(1)
            ed_dest_location = self.ed_flights()['DestLocation'].head(1)

            assert_pandas_eland_series_equal(pd_dest_location, ed_dest_location)
        else:
            # NOOP
            assert True

    def test_num_rows_to_string(self):
        if PY36:
            # check setup works
            assert pd.get_option('display.max_rows') == 60

            # Test eland.DataFrame.to_string vs pandas.DataFrame.to_string
            # In pandas calling 'to_string' without max_rows set, will dump ALL rows

            # Test n-1, n, n+1 for edge cases
            self.num_rows_to_string(DEFAULT_NUM_ROWS_DISPLAYED - 1)
            self.num_rows_to_string(DEFAULT_NUM_ROWS_DISPLAYED)
            with pytest.warns(UserWarning):
                # UserWarning displayed by eland here (compare to pandas with max_rows set)
                self.num_rows_to_string(DEFAULT_NUM_ROWS_DISPLAYED + 1, None, DEFAULT_NUM_ROWS_DISPLAYED)

            # Test for where max_rows lt or gt num_rows
            self.num_rows_to_string(10, 5, 5)
            self.num_rows_to_string(100, 200, 200)
        else:
            # NOOP
            assert True

    def num_rows_to_string(self, rows, max_rows_eland=None, max_rows_pandas=None):
        ed_flights = self.ed_flights()[['DestLocation', 'OriginLocation']]
        pd_flights = self.pd_flights()[['DestLocation', 'OriginLocation']]

        ed_head = ed_flights.head(rows)
        pd_head = pd_flights.head(rows)

        ed_head_str = ed_head.to_string(max_rows=max_rows_eland)
        pd_head_str = pd_head.to_string(max_rows=max_rows_pandas)

        # print("\n", ed_head_str)
        # print("\n", pd_head_str)

        assert pd_head_str == ed_head_str

    def test_empty_dataframe_string(self):
        ed_ecom = self.ed_ecommerce()
        pd_ecom = self.pd_ecommerce()

        ed_ecom_s = ed_ecom[ed_ecom['currency'] == 'USD'].to_string()
        pd_ecom_s = pd_ecom[pd_ecom['currency'] == 'USD'].to_string()

        assert ed_ecom_s == pd_ecom_s

    """
    repr
    """

    def test_num_rows_repr(self):
        self.num_rows_repr(pd.get_option('display.max_rows') - 1, pd.get_option('display.max_rows') - 1)
        self.num_rows_repr(pd.get_option('display.max_rows'), pd.get_option('display.max_rows'))
        self.num_rows_repr(pd.get_option('display.max_rows') + 1, pd.get_option('display.min_rows'))

    def num_rows_repr(self, rows, num_rows_printed):
        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()

        ed_head = ed_flights.head(rows)
        pd_head = pd_flights.head(rows)

        ed_head_str = repr(ed_head)
        pd_head_str = repr(pd_head)

        if num_rows_printed < rows:
            # add 1 for ellipsis
            num_rows_printed = num_rows_printed + 1

        # number of rows is num_rows_printed + 3 (header, summary)
        assert (num_rows_printed + 3) == len(ed_head_str.splitlines())

        assert pd_head_str == ed_head_str

    def test_empty_dataframe_repr(self):
        ed_ecom = self.ed_ecommerce()
        pd_ecom = self.pd_ecommerce()

        ed_ecom_r = repr(ed_ecom[ed_ecom['currency'] == 'USD'])
        pd_ecom_r = repr(pd_ecom[pd_ecom['currency'] == 'USD'])

        assert ed_ecom_r == pd_ecom_r

    """
    to_html
    """

    def test_num_rows_to_html(self):
        # check setup works
        assert pd.get_option('display.max_rows') == 60

        # Test eland.DataFrame.to_string vs pandas.DataFrame.to_string
        # In pandas calling 'to_string' without max_rows set, will dump ALL rows

        # Test n-1, n, n+1 for edge cases
        self.num_rows_to_html(DEFAULT_NUM_ROWS_DISPLAYED - 1)
        self.num_rows_to_html(DEFAULT_NUM_ROWS_DISPLAYED)
        with pytest.warns(UserWarning):
            # UserWarning displayed by eland here
            self.num_rows_to_html(DEFAULT_NUM_ROWS_DISPLAYED + 1, None, DEFAULT_NUM_ROWS_DISPLAYED)

        # Test for where max_rows lt or gt num_rows
        self.num_rows_to_html(10, 5, 5)
        self.num_rows_to_html(100, 200, 200)

    def num_rows_to_html(self, rows, max_rows_eland=None, max_rows_pandas=None):
        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()

        ed_head = ed_flights.head(rows)
        pd_head = pd_flights.head(rows)

        ed_head_str = ed_head.to_html(max_rows=max_rows_eland)
        pd_head_str = pd_head.to_html(max_rows=max_rows_pandas)

        # print(ed_head_str)
        # print(pd_head_str)

        assert pd_head_str == ed_head_str

    def test_empty_dataframe_to_html(self):
        ed_ecom = self.ed_ecommerce()
        pd_ecom = self.pd_ecommerce()

        ed_ecom_h = ed_ecom[ed_ecom['currency'] == 'USD'].to_html()
        pd_ecom_h = pd_ecom[pd_ecom['currency'] == 'USD'].to_html()

        assert ed_ecom_h == pd_ecom_h

    """
    _repr_html_
    """

    def test_num_rows_repr_html(self):
        # check setup works
        assert pd.get_option('display.max_rows') == 60

        show_dimensions = pd.get_option('display.show_dimensions')

        # TODO - there is a bug in 'show_dimensions' as it gets added after the last </div>
        # For now test without this
        pd.set_option('display.show_dimensions', False)

        # Test eland.DataFrame.to_string vs pandas.DataFrame.to_string
        # In pandas calling 'to_string' without max_rows set, will dump ALL rows

        # Test n-1, n, n+1 for edge cases
        self.num_rows_repr_html(pd.get_option('display.max_rows') - 1)
        self.num_rows_repr_html(pd.get_option('display.max_rows'))
        self.num_rows_repr_html(pd.get_option('display.max_rows') + 1, pd.get_option('display.max_rows'))

        # Restore default
        pd.set_option('display.show_dimensions', show_dimensions)

    def num_rows_repr_html(self, rows, max_rows=None):
        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()

        ed_head = ed_flights.head(rows)
        pd_head = pd_flights.head(rows)

        ed_head_str = ed_head._repr_html_()
        pd_head_str = pd_head._repr_html_()

        # print(ed_head_str)
        # print(pd_head_str)

        # Currently pandas display bold_rows=True with >=PY36 and bold_rows=False with 3.5
        # TODO - fix this test for 3.5
        if PY36:
            assert pd_head_str == ed_head_str

    def test_empty_dataframe_repr_html(self):
        # TODO - there is a bug in 'show_dimensions' as it gets added after the last </div>
        # For now test without this
        show_dimensions = pd.get_option('display.show_dimensions')
        pd.set_option('display.show_dimensions', False)

        ed_ecom = self.ed_ecommerce()
        pd_ecom = self.pd_ecommerce()

        ed_ecom_rh = ed_ecom[ed_ecom['currency'] == 'USD']._repr_html_()
        pd_ecom_rh = pd_ecom[pd_ecom['currency'] == 'USD']._repr_html_()

        # Restore default
        pd.set_option('display.show_dimensions', show_dimensions)

        assert ed_ecom_rh == pd_ecom_rh
