# File called _pytest for PyCharm compatability

import pytest

import pandas as pd

from eland.tests.common import TestData

from eland.dataframe import DEFAULT_NUM_ROWS_DISPLAYED


class TestDataFrameRepr(TestData):

    @classmethod
    def setup_class(cls):
        # conftest.py changes this default - restore to original setting
        pd.set_option('display.max_rows', 60)

    """
    to_string
    """
    def test_num_rows_to_string(self):
        # check setup works
        assert pd.get_option('display.max_rows') == 60

        # Test eland.DataFrame.to_string vs pandas.DataFrame.to_string
        # In pandas calling 'to_string' without max_rows set, will dump ALL rows

        # Test n-1, n, n+1 for edge cases
        self.num_rows_to_string(DEFAULT_NUM_ROWS_DISPLAYED-1)
        self.num_rows_to_string(DEFAULT_NUM_ROWS_DISPLAYED)
        with pytest.warns(UserWarning):
            # UserWarning displayed by eland here
            self.num_rows_to_string(DEFAULT_NUM_ROWS_DISPLAYED+1, DEFAULT_NUM_ROWS_DISPLAYED)

    def num_rows_to_string(self, rows, max_rows=None):
        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()

        ed_head = ed_flights.head(rows)
        pd_head = pd_flights.head(rows)

        ed_head_str = ed_head.to_string()
        pd_head_str = pd_head.to_string(max_rows=max_rows)

        #print(ed_head_str)
        #print(pd_head_str)

        assert pd_head_str == ed_head_str

    """
    repr
    """
    def test_num_rows_repr(self):
        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()

        self.num_rows_repr(pd.get_option('display.max_rows')-1, pd.get_option('display.max_rows')-1)
        self.num_rows_repr(pd.get_option('display.max_rows'), pd.get_option('display.max_rows'))
        self.num_rows_repr(pd.get_option('display.max_rows')+1, pd.get_option('display.min_rows'))

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
        assert (num_rows_printed+3) == len(ed_head_str.splitlines())

        assert pd_head_str == ed_head_str

    """
    to_html 
    """
    def test_num_rows_to_html(self):
        # check setup works
        assert pd.get_option('display.max_rows') == 60

        # Test eland.DataFrame.to_string vs pandas.DataFrame.to_string
        # In pandas calling 'to_string' without max_rows set, will dump ALL rows

        # Test n-1, n, n+1 for edge cases
        self.num_rows_to_html(DEFAULT_NUM_ROWS_DISPLAYED-1)
        self.num_rows_to_html(DEFAULT_NUM_ROWS_DISPLAYED)
        with pytest.warns(UserWarning):
            # UserWarning displayed by eland here
            self.num_rows_to_html(DEFAULT_NUM_ROWS_DISPLAYED+1, DEFAULT_NUM_ROWS_DISPLAYED)

    def num_rows_to_html(self, rows, max_rows=None):
        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()

        ed_head = ed_flights.head(rows)
        pd_head = pd_flights.head(rows)

        ed_head_str = ed_head.to_html()
        pd_head_str = pd_head.to_html(max_rows=max_rows)

        print(ed_head_str)
        print(pd_head_str)

        assert pd_head_str == ed_head_str


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
        self.num_rows_repr_html(pd.get_option('display.max_rows')-1)
        self.num_rows_repr_html(pd.get_option('display.max_rows'))
        self.num_rows_repr_html(pd.get_option('display.max_rows')+1, pd.get_option('display.max_rows'))

        # Restore default
        pd.set_option('display.show_dimensions', show_dimensions)

    def num_rows_repr_html(self, rows, max_rows=None):
        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()

        ed_head = ed_flights.head(rows)
        pd_head = pd_flights.head(rows)

        ed_head_str = ed_head._repr_html_()
        pd_head_str = pd_head._repr_html_()

        #print(ed_head_str)
        #print(pd_head_str)

        assert pd_head_str == ed_head_str
