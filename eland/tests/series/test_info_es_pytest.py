# File called _pytest for PyCharm compatability

from pandas.util.testing import assert_almost_equal

from eland.tests.common import TestData

import eland as ed


class TestSeriesInfoEs(TestData):

    def test_flights_info_es(self):
        ed_flights = self.ed_flights()['AvgTicketPrice']

        # No assertion, just test it can be called
        info_es = ed_flights.info_es()

