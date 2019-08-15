# File called _pytest for PyCharm compatability
import pandas as pd
import eland as ed

from eland.tests.common import TestData
from eland.tests.common import (
    assert_pandas_eland_frame_equal,
    assert_pandas_eland_series_equal
)

import numpy as np

class TestDataFrameNUnique(TestData):

    def test_nunique1(self):
        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()

        print(pd_flights.dtypes)
        print(ed_flights.dtypes)
        print(ed_flights.nunique())

