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

from pandas.util.testing import assert_series_equal

from eland.tests.common import TestData


class TestDataFrameMetrics(TestData):
    funcs = ['max', 'min', 'mean', 'sum']

    def test_flights_metrics(self):
        pd_flights = self.pd_flights()
        ed_flights = self.ed_flights()

        for func in self.funcs:
            pd_metric = getattr(pd_flights, func)(numeric_only=True)
            ed_metric = getattr(ed_flights, func)(numeric_only=True)

            assert_series_equal(pd_metric, ed_metric)

    def test_ecommerce_selected_non_numeric_source_fields(self):
        # None of these are numeric
        columns = ['category', 'currency', 'customer_birth_date', 'customer_first_name', 'user']

        pd_ecommerce = self.pd_ecommerce()[columns]
        ed_ecommerce = self.ed_ecommerce()[columns]

        for func in self.funcs:
            assert_series_equal(getattr(pd_ecommerce, func)(numeric_only=True),
                                getattr(ed_ecommerce, func)(numeric_only=True),
                                check_less_precise=True)

    def test_ecommerce_selected_mixed_numeric_source_fields(self):
        # Some of these are numeric
        columns = ['category', 'currency', 'taxless_total_price', 'customer_birth_date',
                   'total_quantity', 'customer_first_name', 'user']

        pd_ecommerce = self.pd_ecommerce()[columns]
        ed_ecommerce = self.ed_ecommerce()[columns]

        for func in self.funcs:
            assert_series_equal(getattr(pd_ecommerce, func)(numeric_only=True),
                                getattr(ed_ecommerce, func)(numeric_only=True),
                                check_less_precise=True)

    def test_ecommerce_selected_all_numeric_source_fields(self):
        # All of these are numeric
        columns = ['total_quantity', 'taxful_total_price', 'taxless_total_price']

        pd_ecommerce = self.pd_ecommerce()[columns]
        ed_ecommerce = self.ed_ecommerce()[columns]

        for func in self.funcs:
            assert_series_equal(getattr(pd_ecommerce, func)(numeric_only=True),
                                getattr(ed_ecommerce, func)(numeric_only=True),
                                check_less_precise=True)
