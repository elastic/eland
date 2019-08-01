# File called _pytest for PyCharm compatability

import numpy as np
import pandas as pd

from eland.tests.common import TestData


class TestDataFrameAggs(TestData):

    def test_to_aggs1(self):
        pd_flights = self.pd_flights()
        ed_flights = self.ed_flights()

        pd_sum_min = pd_flights.select_dtypes(include=[np.number]).agg(['sum', 'min'])
        print(type(pd_sum_min))
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(pd_sum_min)

        ed_sum_min = ed_flights.select_dtypes(include=[np.number]).agg(['sum', 'min'])
        print(type(ed_sum_min))
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(ed_sum_min)
