# File called _pytest for PyCharm compatability

from eland.tests.common import TestData

from pandas.util.testing import assert_series_equal

import numpy as np

class TestDataFrameHist(TestData):

    def test_dataframe_hist1(self):
        test_data = TestData()

        pd_flights = test_data.pd_flights()[['DistanceKilometers', 'DistanceMiles', 'FlightDelayMin', 'FlightTimeHour']]
        ed_flights = test_data.ed_flights()[['DistanceKilometers', 'DistanceMiles', 'FlightDelayMin', 'FlightTimeHour']]

        pd_flights.hist(figsize=[10, 10])
        ed_flights.hist(figsize=[10, 10])

        pd_min = pd_flights['DistanceKilometers'].min()
        pd_max = pd_flights['DistanceKilometers'].max()

        ed_min = ed_flights['DistanceKilometers'].min()
        ed_max = ed_flights['DistanceKilometers'].max()

        num_bins = 10.0

        bins = np.linspace(ed_min, ed_max, num=num_bins+1)

        print(bins)

        print(np.diff(bins).mean())

        hist = ed_flights['DistanceKilometers'].hist(np.diff(bins).mean())


        x = [2956.,  768.,  719., 2662., 2934., 1320.,  641.,  529.,  426.,  104.]
        bins = [0., 1988.14823146, 3976.29646292, 5964.44469437, 7952.59292583, 9940.74115729, 11928.88938875, 13917.03762021, 15905.18585166,17893.33408312,19881.48231458]

        print(len(x))
        print(len(bins))

        a = bins[0:10]

        print(np.histogram(a, weights=x, bins=bins))

