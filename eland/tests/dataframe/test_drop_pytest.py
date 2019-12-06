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
    assert_pandas_eland_frame_equal
)


class TestDataFrameDrop(TestData):

    def test_flights_small_drop(self):
        ed_flights_small = self.ed_flights_small()
        pd_flights_small = self.pd_flights_small()

        # ['AvgTicketPrice', 'Cancelled', 'Carrier', 'Dest', 'DestAirportID',
        #        'DestCityName', 'DestCountry', 'DestLocation', 'DestRegion',
        #        'DestWeather', 'DistanceKilometers', 'DistanceMiles', 'FlightDelay',
        #        'FlightDelayMin', 'FlightDelayType', 'FlightNum', 'FlightTimeHour',
        #        'FlightTimeMin', 'Origin', 'OriginAirportID', 'OriginCityName',
        #        'OriginCountry', 'OriginLocation', 'OriginRegion', 'OriginWeather',
        #        'dayOfWeek', 'timestamp']
        pd_col0 = pd_flights_small.drop(['Carrier', 'DestCityName'], axis=1)
        pd_col1 = pd_flights_small.drop(columns=['Carrier', 'DestCityName'])

        ed_col0 = ed_flights_small.drop(['Carrier', 'DestCityName'], axis=1)
        ed_col1 = ed_flights_small.drop(columns=['Carrier', 'DestCityName'])

        assert_pandas_eland_frame_equal(pd_col0, ed_col0)
        assert_pandas_eland_frame_equal(pd_col1, ed_col1)

        # Drop rows by index
        pd_idx0 = pd_flights_small.drop(['1', '2'])
        ed_idx0 = ed_flights_small.drop(['1', '2'])

        assert_pandas_eland_frame_equal(pd_idx0, ed_idx0)
