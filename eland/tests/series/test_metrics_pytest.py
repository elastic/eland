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

from pandas.util.testing import assert_almost_equal

from eland.tests.common import TestData


class TestSeriesMetrics(TestData):
    funcs = ['max', 'min', 'mean', 'sum']
    timestamp_funcs = ['max', 'min', 'mean']

    def test_flights_metrics(self):
        pd_flights = self.pd_flights()['AvgTicketPrice']
        ed_flights = self.ed_flights()['AvgTicketPrice']

        for func in self.funcs:
            pd_metric = getattr(pd_flights, func)()
            ed_metric = getattr(ed_flights, func)()
            assert_almost_equal(pd_metric, ed_metric, check_less_precise=True)

    def test_flights_timestamp(self):
        pd_flights = self.pd_flights()['timestamp']
        ed_flights = self.ed_flights()['timestamp']

        for func in self.timestamp_funcs:
            pd_metric = getattr(pd_flights, func)()
            ed_metric = getattr(ed_flights, func)()
            pd_metric = pd_metric.floor("S") # floor or pandas mean with have ns
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
