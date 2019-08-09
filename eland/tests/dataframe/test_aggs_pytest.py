# File called _pytest for PyCharm compatability

import numpy as np
import pandas as pd

from eland.tests.common import TestData


class TestDataFrameAggs(TestData):

    def test_to_aggs1(self):
        pd_flights = self.pd_flights()
        ed_flights = self.ed_flights()

        pd_numerics = pd_flights.select_dtypes(include=[np.number])
        print(pd_numerics.columns)
        print(pd_numerics.agg('abs')) # all rows
        print(pd_numerics.agg('all')) # columns True/False
        print(pd_numerics.agg('any')) # columns True/False
        print(pd_numerics.agg('corr')) # matrix col/col
        print(pd_numerics.agg('count')) # columns count
        print(pd_numerics.agg('cov')) # matrix col/col
        print(pd_numerics.agg('cummax')) # all rows
        print(pd_numerics.agg('cummin')) # all rows
        print(pd_numerics.agg('cumprod')) # all rows
        print(pd_numerics.agg('cumsum')) # all rows
        print(pd_numerics.agg('describe')) # describe
        print(pd_numerics.agg('diff'))  # all rows
        print(pd_numerics.agg('kurt')) # ?>
        print(pd_numerics.agg('mad')) # col
        print('MAX')
        print(pd_numerics.agg('max')) # col
        print(pd_numerics.agg('mean')) # col
        print(pd_numerics.agg('median')) # col
        print(pd_numerics.agg('min')) # col
        print(pd_numerics.agg('mode')) # col
        print(pd_numerics.agg('pct_change')) # all rows
        print(pd_numerics.agg('prod')) # all rows
        print(pd_numerics.agg('quantile')) # col
        print(pd_numerics.agg('rank')) # col
        print(pd_numerics.agg('round')) # all rows
        print('SEM')
        print(pd_numerics.agg('sem')) # col
        print(pd_numerics.agg('skew')) # col
        print(pd_numerics.agg('sum')) # col
        print(pd_numerics.agg('std')) # col
        print(pd_numerics.agg('var')) # col
        print(pd_numerics.agg('nunique')) # col

        print(pd_numerics.aggs(np.sqrt)) # all rows


        return

        pd_sum_min = pd_flights.select_dtypes(include=[np.number]).agg(['sum', 'min'])
        print(type(pd_sum_min))
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(pd_sum_min)

        ed_sum_min = ed_flights.select_dtypes(include=[np.number]).agg(['sum', 'min'])
        print(type(ed_sum_min))
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(ed_sum_min)
