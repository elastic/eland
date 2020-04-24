# Licensed to Elasticsearch B.V under one or more agreements.
# Elasticsearch B.V licenses this file to you under the Apache 2.0 License.
# See the LICENSE file in the project root for more information

# File called _pytest for PyCharm compatability

from eland.tests.common import TestData
from eland.tests.common import assert_pandas_eland_frame_equal


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
        pd_col0 = pd_flights_small.drop(["Carrier", "DestCityName"], axis=1)
        pd_col1 = pd_flights_small.drop(columns=["Carrier", "DestCityName"])

        ed_col0 = ed_flights_small.drop(["Carrier", "DestCityName"], axis=1)
        ed_col1 = ed_flights_small.drop(columns=["Carrier", "DestCityName"])

        assert_pandas_eland_frame_equal(pd_col0, ed_col0)
        assert_pandas_eland_frame_equal(pd_col1, ed_col1)

        # Drop rows by index
        pd_idx0 = pd_flights_small.drop(["1", "2"])
        ed_idx0 = ed_flights_small.drop(["1", "2"])

        assert_pandas_eland_frame_equal(pd_idx0, ed_idx0)
