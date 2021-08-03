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
from pandas.testing import assert_frame_equal

from tests.common import TestData


class TestDataFrameHist(TestData):
    def test_flights_hist(self):
        pd_flights = self.pd_flights()
        ed_flights = self.ed_flights()

        num_bins = 10

        # pandas data
        pd_distancekilometers = np.histogram(pd_flights["DistanceKilometers"], num_bins)
        pd_flightdelaymin = np.histogram(pd_flights["FlightDelayMin"], num_bins)

        pd_bins = pd.DataFrame(
            {
                "DistanceKilometers": pd_distancekilometers[1],
                "FlightDelayMin": pd_flightdelaymin[1],
            }
        )
        pd_weights = pd.DataFrame(
            {
                "DistanceKilometers": pd_distancekilometers[0],
                "FlightDelayMin": pd_flightdelaymin[0],
            }
        )

        _ = ed_flights[["DistanceKilometers", "FlightDelayMin"]]

        ed_bins, ed_weights = ed_flights[
            ["DistanceKilometers", "FlightDelayMin"]
        ]._hist(num_bins=num_bins)

        # Numbers are slightly different
        assert_frame_equal(pd_bins, ed_bins, check_exact=False)
        assert_frame_equal(pd_weights, ed_weights, check_exact=False)

    def test_flights_filtered_hist(self):
        pd_flights = self.pd_flights()
        ed_flights = self.ed_flights()

        pd_flights = pd_flights[pd_flights.FlightDelayMin > 0]
        ed_flights = ed_flights[ed_flights.FlightDelayMin > 0]

        num_bins = 10

        # pandas data
        pd_distancekilometers = np.histogram(pd_flights["DistanceKilometers"], num_bins)
        pd_flightdelaymin = np.histogram(pd_flights["FlightDelayMin"], num_bins)

        pd_bins = pd.DataFrame(
            {
                "DistanceKilometers": pd_distancekilometers[1],
                "FlightDelayMin": pd_flightdelaymin[1],
            }
        )
        pd_weights = pd.DataFrame(
            {
                "DistanceKilometers": pd_distancekilometers[0],
                "FlightDelayMin": pd_flightdelaymin[0],
            }
        )

        _ = ed_flights[["DistanceKilometers", "FlightDelayMin"]]

        ed_bins, ed_weights = ed_flights[
            ["DistanceKilometers", "FlightDelayMin"]
        ]._hist(num_bins=num_bins)

        # Numbers are slightly different
        assert_frame_equal(pd_bins, ed_bins, check_exact=False)
        assert_frame_equal(pd_weights, ed_weights, check_exact=False)
