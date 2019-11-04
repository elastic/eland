# File called _pytest for PyCharm compatability
import numpy as np

from eland.tests.common import TestData
from eland.tests.common import (
    assert_pandas_eland_frame_equal
)


class TestDataFrameSelectDTypes(TestData):

    def test_select_dtypes_include_number(self):
        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()

        ed_flights_numeric = ed_flights.select_dtypes(include=[np.number])
        pd_flights_numeric = pd_flights.select_dtypes(include=[np.number])

        assert_pandas_eland_frame_equal(pd_flights_numeric.head(103), ed_flights_numeric.head(103))

    def test_select_dtypes_exclude_number(self):
        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()

        ed_flights_non_numeric = ed_flights.select_dtypes(exclude=[np.number])
        pd_flights_non_numeric = pd_flights.select_dtypes(exclude=[np.number])

        assert_pandas_eland_frame_equal(pd_flights_non_numeric.head(103), ed_flights_non_numeric.head(103))
