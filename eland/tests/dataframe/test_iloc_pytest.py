# File called _pytest for PyCharm compatability
import pandas as pd
import eland as ed

from eland.tests.common import TestData
from eland.tests.common import (
    assert_pandas_eland_frame_equal,
    assert_pandas_eland_series_equal
)

import numpy as np

class TestDataFrameiLoc(TestData):

    def test_iloc1(self):
        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()

        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iloc.html#pandas.DataFrame.iloc

        #pd_flights.info()

        pd_iloc0 = pd_flights.iloc[0]
        pd_iloc1= pd_flights.iloc[[0]]
        pd_iloc2= pd_flights.iloc[[0, 1]]
        pd_iloc3 = pd_flights.iloc[:3]
        pd_iloc5 = pd_flights.iloc[0, 1]
        pd_iloc6 = pd_flights.iloc[[0, 2], [1, 3]]
        pd_iloc7 = pd_flights.iloc[1:3, 0:3]

        ed_iloc0 = ed_flights.iloc[0]
        ed_iloc1 = ed_flights.iloc[[0]]
        ed_iloc2 = ed_flights.iloc[[0, 1]]
        ed_iloc3 = ed_flights.iloc[:3]
        ed_iloc5 = ed_flights.iloc[0, 1]
        ed_iloc6 = ed_flights.iloc[[0, 2], [1, 3]]
        ed_iloc7 = ed_flights.iloc[1:3, 0:3]

        #assert_pandas_eland_frame_equal(pd_iloc0, ed_iloc0) # pd_iloc0 is Series
        assert_pandas_eland_frame_equal(pd_iloc1, ed_iloc1)
        assert_pandas_eland_frame_equal(pd_iloc2, ed_iloc2)
        assert_pandas_eland_frame_equal(pd_iloc3, ed_iloc3)
        #assert_pandas_eland_frame_equal(pd_iloc5, ed_iloc5) # pd_iloc5 is numpy_bool
        assert_pandas_eland_frame_equal(pd_iloc6, ed_iloc6)
        assert_pandas_eland_frame_equal(pd_iloc7, ed_iloc7)
