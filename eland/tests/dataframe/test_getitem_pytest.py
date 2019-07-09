# File called _pytest for PyCharm compatability
import pandas as pd

from eland.tests.common import TestData
from eland.tests.common import (
    assert_pandas_eland_frame_equal,
    assert_pandas_eland_series_equal
)



class TestDataFrameGetItem(TestData):

    def test_getitem1(self):
        ed_flights = self.ed_flights().head(103)
        pd_flights = self.pd_flights().head(103)

        ed_flights_OriginAirportID = ed_flights['OriginAirportID']
        pd_flights_OriginAirportID = pd_flights['OriginAirportID']

        assert_pandas_eland_series_equal(pd_flights_OriginAirportID, ed_flights_OriginAirportID)

    def test_getitem2(self):
        ed_flights = self.ed_flights().head(42)
        pd_flights = self.pd_flights().head(42)

        ed_flights_slice = ed_flights[['OriginAirportID', 'AvgTicketPrice', 'Carrier']]
        pd_flights_slice = pd_flights[['OriginAirportID', 'AvgTicketPrice', 'Carrier']]

        assert_pandas_eland_frame_equal(pd_flights_slice, ed_flights_slice)

    def test_getitem3(self):
        ed_flights = self.ed_flights().head(89)
        pd_flights = self.pd_flights().head(89)

        ed_flights_OriginAirportID = ed_flights.OriginAirportID
        pd_flights_OriginAirportID = pd_flights.OriginAirportID

        assert_pandas_eland_series_equal(pd_flights_OriginAirportID, ed_flights_OriginAirportID)
