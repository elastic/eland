#  Copyright 2019 Elasticsearch BV
#
#      Licensed under the Apache License, Version 2.0 (the "License");
#      you may not use this file except in compliance with the License.
#      You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#      Unless required by applicable law or agreed to in writing, software
#      distributed under the License is distributed on an "AS IS" BASIS,
#      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#      See the License for the specific language governing permissions and
#      limitations under the License.

# File called _pytest for PyCharm compatability

from eland.tests.common import TestData
from eland.tests.common import (
    assert_pandas_eland_frame_equal,
    assert_pandas_eland_series_equal
)


class TestDataFrameGetItem(TestData):

    def test_getitem_one_attribute(self):
        ed_flights = self.ed_flights().head(103)
        pd_flights = self.pd_flights().head(103)

        ed_flights_OriginAirportID = ed_flights['OriginAirportID']
        pd_flights_OriginAirportID = pd_flights['OriginAirportID']

        assert_pandas_eland_series_equal(pd_flights_OriginAirportID, ed_flights_OriginAirportID)

    def test_getitem_attribute_list(self):
        ed_flights = self.ed_flights().head(42)
        pd_flights = self.pd_flights().head(42)

        ed_flights_slice = ed_flights[['OriginAirportID', 'AvgTicketPrice', 'Carrier']]
        pd_flights_slice = pd_flights[['OriginAirportID', 'AvgTicketPrice', 'Carrier']]

        assert_pandas_eland_frame_equal(pd_flights_slice, ed_flights_slice)

    def test_getitem_one_argument(self):
        ed_flights = self.ed_flights().head(89)
        pd_flights = self.pd_flights().head(89)

        ed_flights_OriginAirportID = ed_flights.OriginAirportID
        pd_flights_OriginAirportID = pd_flights.OriginAirportID

        assert_pandas_eland_series_equal(pd_flights_OriginAirportID, ed_flights_OriginAirportID)

    def test_getitem_multiple_calls(self):
        ed_flights = self.ed_flights().head(89)
        pd_flights = self.pd_flights().head(89)

        ed_col0 = ed_flights[['DestCityName', 'DestCountry', 'DestLocation', 'DestRegion']]
        try:
            ed_col1 = ed_col0['Carrier']
        except KeyError:
            pass

        pd_col1 = pd_flights['DestCountry']
        ed_col1 = ed_col0['DestCountry']

        assert_pandas_eland_series_equal(pd_col1, ed_col1)
