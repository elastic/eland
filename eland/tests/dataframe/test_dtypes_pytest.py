# File called _pytest for PyCharm compatability

import numpy as np

from pandas.util.testing import assert_series_equal

from eland.tests.common import TestData
from eland.tests.common import assert_pandas_eland_frame_equal


class TestDataFrameDtypes(TestData):

    def test_flights_dtypes(self):
        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()

        assert_series_equal(pd_flights.dtypes, ed_flights.dtypes)

    def test_flights_select_dtypes(self):
        ed_flights = self.ed_flights_small()
        pd_flights = self.pd_flights_small()

        assert_pandas_eland_frame_equal(
            pd_flights.select_dtypes(include=np.number),
            ed_flights.select_dtypes(include=np.number)
        )
