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
from pandas.util.testing import assert_almost_equal

from eland.tests.common import TestData


class TestDataFrameAggs(TestData):

    def test_basic_aggs(self):
        pd_flights = self.pd_flights()
        ed_flights = self.ed_flights()

        pd_sum_min = pd_flights.select_dtypes(include=[np.number]).agg(['sum', 'min'])
        ed_sum_min = ed_flights.select_dtypes(include=[np.number]).agg(['sum', 'min'])

        # Eland returns all float values for all metric aggs, pandas can return int
        # TODO - investigate this more
        pd_sum_min = pd_sum_min.astype('float64')
        assert_almost_equal(pd_sum_min, ed_sum_min)

        pd_sum_min_std = pd_flights.select_dtypes(include=[np.number]).agg(['sum', 'min', 'std'])
        ed_sum_min_std = ed_flights.select_dtypes(include=[np.number]).agg(['sum', 'min', 'std'])

        print(pd_sum_min_std.dtypes)
        print(ed_sum_min_std.dtypes)

        assert_almost_equal(pd_sum_min_std, ed_sum_min_std, check_less_precise=True)

    def test_terms_aggs(self):
        pd_flights = self.pd_flights()
        ed_flights = self.ed_flights()

        pd_sum_min = pd_flights.select_dtypes(include=[np.number]).agg(['sum', 'min'])
        ed_sum_min = ed_flights.select_dtypes(include=[np.number]).agg(['sum', 'min'])

        # Eland returns all float values for all metric aggs, pandas can return int
        # TODO - investigate this more
        pd_sum_min = pd_sum_min.astype('float64')
        assert_almost_equal(pd_sum_min, ed_sum_min)

        pd_sum_min_std = pd_flights.select_dtypes(include=[np.number]).agg(['sum', 'min', 'std'])
        ed_sum_min_std = ed_flights.select_dtypes(include=[np.number]).agg(['sum', 'min', 'std'])

        print(pd_sum_min_std.dtypes)
        print(ed_sum_min_std.dtypes)

        assert_almost_equal(pd_sum_min_std, ed_sum_min_std, check_less_precise=True)
