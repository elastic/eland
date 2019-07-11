# File called _pytest for PyCharm compatability
import pandas as pd
import eland as ed

from eland.tests.common import TestData
from eland.tests.common import (
    assert_eland_frame_equal,
    assert_pandas_eland_frame_equal,
    assert_pandas_eland_series_equal
)

import numpy as np

class TestDataFrameDrop(TestData):

    def test_drop1(self):
        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()

        # ['AvgTicketPrice', 'Cancelled', 'Carrier', 'Dest', 'DestAirportID',
        #        'DestCityName', 'DestCountry', 'DestLocation', 'DestRegion',
        #        'DestWeather', 'DistanceKilometers', 'DistanceMiles', 'FlightDelay',
        #        'FlightDelayMin', 'FlightDelayType', 'FlightNum', 'FlightTimeHour',
        #        'FlightTimeMin', 'Origin', 'OriginAirportID', 'OriginCityName',
        #        'OriginCountry', 'OriginLocation', 'OriginRegion', 'OriginWeather',
        #        'dayOfWeek', 'timestamp']
        pd_col0 = pd_flights.drop(['Carrier', 'DestCityName'], axis=1)
        pd_col1 = pd_flights.drop(columns=['Carrier', 'DestCityName'])

        ed_col0 = ed_flights.drop(['Carrier', 'DestCityName'], axis=1)
        ed_col1 = ed_flights.drop(columns=['Carrier', 'DestCityName'])

        #assert_pandas_eland_frame_equal(pd_col0, ed_col0)
        #assert_pandas_eland_frame_equal(pd_col1, ed_col1)

        # Drop rows by index
        pd_idx0 = pd_flights.drop(['1', '2'])
        ed_idx0 = ed_flights.drop(['1', '2'])

        print(pd_idx0.info())
        print(ed_idx0.info())

        assert_pandas_eland_frame_equal(pd_idx0, ed_idx0)

        """
        #assert_pandas_eland_frame_equal(pd_iloc0, ed_iloc0) # pd_iloc0 is Series
        assert_pandas_eland_frame_equal(pd_iloc1, ed_iloc1)
        assert_pandas_eland_frame_equal(pd_iloc2, ed_iloc2)
        assert_pandas_eland_frame_equal(pd_iloc3, ed_iloc3)
        assert_pandas_eland_frame_equal(pd_iloc4, ed_iloc4)
        #assert_pandas_eland_frame_equal(pd_iloc5, ed_iloc5) # pd_iloc5 is numpy_bool
        assert_pandas_eland_frame_equal(pd_iloc6, ed_iloc6)
        assert_pandas_eland_frame_equal(pd_iloc7, ed_iloc7)
        assert_pandas_eland_frame_equal(pd_iloc8, ed_iloc8)
        assert_pandas_eland_frame_equal(pd_iloc9, ed_iloc9)
        """
