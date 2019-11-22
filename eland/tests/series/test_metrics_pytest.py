# File called _pytest for PyCharm compatability

from pandas.util.testing import assert_almost_equal

from eland.tests.common import TestData

import eland as ed


class TestSeriesMetrics(TestData):

    funcs = ['max', 'min', 'mean', 'sum']

    def test_flights_metrics(self):
        pd_flights = self.pd_flights()['AvgTicketPrice']
        ed_flights = self.ed_flights()['AvgTicketPrice']

        for func in self.funcs:
            pd_metric = getattr(pd_flights, func)()
            ed_metric = getattr(ed_flights, func)()
            assert_almost_equal(pd_metric, ed_metric, check_less_precise=True)

    def test_ecommerce_selected_non_numeric_source_fields(self):
        # None of these are numeric
        column = 'category'

        ed_ecommerce = self.ed_ecommerce()[column]

        for func in self.funcs:
            ed_metric = getattr(ed_ecommerce, func)()
            assert ed_metric.empty


    def test_ecommerce_selected_all_numeric_source_fields(self):
        # All of these are numeric
        columns = ['total_quantity', 'taxful_total_price', 'taxless_total_price']

        for column in columns:
            pd_ecommerce = self.pd_ecommerce()[column]
            ed_ecommerce = self.ed_ecommerce()[column]

            for func in self.funcs:
                assert_almost_equal(getattr(pd_ecommerce, func)(), getattr(ed_ecommerce, func)(),
                                    check_less_precise=True)
