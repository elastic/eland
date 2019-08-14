# File called _pytest for PyCharm compatability

import numpy as np
import pandas as pd
from pandas.util.testing import (assert_almost_equal)

from eland.tests.common import TestData


class TestDataFrameAggs(TestData):

    def test_to_aggs1(self):
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
