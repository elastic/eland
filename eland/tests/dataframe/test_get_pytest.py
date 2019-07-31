# File called _pytest for PyCharm compatability
import pandas as pd
import eland as ed

from eland.tests.common import TestData
from eland.tests.common import (
    assert_pandas_eland_frame_equal,
    assert_pandas_eland_series_equal
)

import numpy as np

class TestDataFrameGet(TestData):

    def test_get1(self):
        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()

        ed_get0 = ed_flights.get('Carrier')
        pd_get0 = pd_flights.get('Carrier')

        print(ed_get0, type(ed_get0))
        print(pd_get0, type(pd_get0))
