# File called _pytest for PyCharm compatability

from eland.tests.common import TestData


class TestSeriesInfoEs(TestData):

    def test_flights_info_es(self):
        ed_flights = self.ed_flights()['AvgTicketPrice']

        # No assertion, just test it can be called
        info_es = ed_flights.info_es()
