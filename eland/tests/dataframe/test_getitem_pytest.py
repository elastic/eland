# Licensed to Elasticsearch B.V under one or more agreements.
# Elasticsearch B.V licenses this file to you under the Apache 2.0 License.
# See the LICENSE file in the project root for more information

# File called _pytest for PyCharm compatability

from eland.tests.common import TestData
from eland.tests.common import (
    assert_pandas_eland_frame_equal,
    assert_pandas_eland_series_equal,
)


class TestDataFrameGetItem(TestData):
    def test_getitem_one_attribute(self):
        ed_flights = self.ed_flights().head(103)
        pd_flights = self.pd_flights().head(103)

        ed_flights_OriginAirportID = ed_flights["OriginAirportID"]
        pd_flights_OriginAirportID = pd_flights["OriginAirportID"]

        assert_pandas_eland_series_equal(
            pd_flights_OriginAirportID, ed_flights_OriginAirportID
        )

    def test_getitem_attribute_list(self):
        ed_flights = self.ed_flights().head(42)
        pd_flights = self.pd_flights().head(42)

        ed_flights_slice = ed_flights[["OriginAirportID", "AvgTicketPrice", "Carrier"]]
        pd_flights_slice = pd_flights[["OriginAirportID", "AvgTicketPrice", "Carrier"]]

        assert_pandas_eland_frame_equal(pd_flights_slice, ed_flights_slice)

    def test_getitem_one_argument(self):
        ed_flights = self.ed_flights().head(89)
        pd_flights = self.pd_flights().head(89)

        ed_flights_OriginAirportID = ed_flights.OriginAirportID
        pd_flights_OriginAirportID = pd_flights.OriginAirportID

        assert_pandas_eland_series_equal(
            pd_flights_OriginAirportID, ed_flights_OriginAirportID
        )

    def test_getitem_multiple_calls(self):
        ed_flights = self.ed_flights().head(89)
        pd_flights = self.pd_flights().head(89)

        ed_col0 = ed_flights[
            ["DestCityName", "DestCountry", "DestLocation", "DestRegion"]
        ]
        try:
            ed_col1 = ed_col0["Carrier"]
        except KeyError:
            pass

        pd_col1 = pd_flights["DestCountry"]
        ed_col1 = ed_col0["DestCountry"]

        assert_pandas_eland_series_equal(pd_col1, ed_col1)
