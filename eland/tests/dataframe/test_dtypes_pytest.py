#  Licensed to Elasticsearch B.V. under one or more contributor
#  license agreements. See the NOTICE file distributed with
#  this work for additional information regarding copyright
#  ownership. Elasticsearch B.V. licenses this file to you under
#  the Apache License, Version 2.0 (the "License"); you may
#  not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
# 	http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the License is distributed on an
#  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
#  specific language governing permissions and limitations
#  under the License.

# File called _pytest for PyCharm compatability

import numpy as np
import pandas as pd

from eland.tests.common import assert_series_equal


class TestDataFrameDtypes:
    def test_dtypes(self, df):
        for i in range(0, len(df.dtypes) - 1):
            assert isinstance(df.dtypes[i], type(df.dtypes[i]))

    def test_select_dtypes(self, df):
        df.select_dtypes(include=np.number)
        df.select_dtypes(exclude=np.number)
        df.select_dtypes(include=np.float64)
        df.select_dtypes(exclude=np.float64)

    def test_es_dtypes(self, testdata):
        df = testdata.ed_flights_small()
        assert_series_equal(
            df.es_dtypes,
            pd.Series(
                {
                    "AvgTicketPrice": "float",
                    "Cancelled": "boolean",
                    "Carrier": "keyword",
                    "Dest": "keyword",
                    "DestAirportID": "keyword",
                    "DestCityName": "keyword",
                    "DestCountry": "keyword",
                    "DestLocation": "geo_point",
                    "DestRegion": "keyword",
                    "DestWeather": "keyword",
                    "DistanceKilometers": "float",
                    "DistanceMiles": "float",
                    "FlightDelay": "boolean",
                    "FlightDelayMin": "integer",
                    "FlightDelayType": "keyword",
                    "FlightNum": "keyword",
                    "FlightTimeHour": "float",
                    "FlightTimeMin": "float",
                    "Origin": "keyword",
                    "OriginAirportID": "keyword",
                    "OriginCityName": "keyword",
                    "OriginCountry": "keyword",
                    "OriginLocation": "geo_point",
                    "OriginRegion": "keyword",
                    "OriginWeather": "keyword",
                    "dayOfWeek": "byte",
                    "timestamp": "date",
                }
            ),
        )
