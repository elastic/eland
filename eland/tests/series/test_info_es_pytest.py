# Licensed to Elasticsearch B.V under one or more agreements.
# Elasticsearch B.V licenses this file to you under the Apache 2.0 License.
# See the LICENSE file in the project root for more information

# File called _pytest for PyCharm compatability

from eland.tests.common import TestData


class TestSeriesInfoEs(TestData):
    def test_flights_info_es(self):
        ed_flights = self.ed_flights()["AvgTicketPrice"]

        # No assertion, just test it can be called
        ed_flights.info_es()
