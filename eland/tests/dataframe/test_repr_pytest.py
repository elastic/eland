# File called _pytest for PyCharm compatability

import pytest

from eland.tests.common import TestData


class TestDataFrameRepr(TestData):

    def test_head_101_to_string(self):
        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()

        ed_head_101 = ed_flights.head(101)
        pd_head_101 = pd_flights.head(101)

        # This sets max_rows=60 by default (but throws userwarning)
        with pytest.warns(UserWarning):
            ed_head_101_str = ed_head_101.to_string()
        pd_head_101_str = pd_head_101.to_string(max_rows=60)

        assert pd_head_101_str  == ed_head_101_str

    def test_head_11_to_string2(self):
        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()

        ed_head_11 = ed_flights.head(11)
        pd_head_11 = pd_flights.head(11)

        ed_head_11_str = ed_head_11.to_string(max_rows=60)
        pd_head_11_str = pd_head_11.to_string(max_rows=60)

        assert pd_head_11_str == ed_head_11_str

    def test_less_than_max_rows_to_string(self):
        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()

        ed_less_than_max = ed_flights[ed_flights['AvgTicketPrice']>1190]
        pd_less_than_max = pd_flights[pd_flights['AvgTicketPrice']>1190]

        ed_less_than_max_str = ed_less_than_max.to_string()
        pd_less_than_max_str = pd_less_than_max.to_string()

    def test_repr(self):
        ed_ecommerce = self.ed_ecommerce()
        pd_ecommerce = self.pd_ecommerce()

        ed_head_18 = ed_ecommerce.head(18)
        pd_head_18 = pd_ecommerce.head(18)

        ed_head_18_repr = repr(ed_head_18)
        pd_head_18_repr = repr(pd_head_18)

        assert ed_head_18_repr == pd_head_18_repr
