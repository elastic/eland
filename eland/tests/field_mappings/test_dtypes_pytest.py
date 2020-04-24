# Licensed to Elasticsearch B.V under one or more agreements.
# Elasticsearch B.V licenses this file to you under the Apache 2.0 License.
# See the LICENSE file in the project root for more information

# File called _pytest for PyCharm compatability
from pandas.testing import assert_series_equal

from eland.field_mappings import FieldMappings
from eland.tests import ES_TEST_CLIENT, FLIGHTS_INDEX_NAME
from eland.tests.common import TestData


class TestDTypes(TestData):
    def test_all_fields(self):
        field_mappings = FieldMappings(
            client=ES_TEST_CLIENT, index_pattern=FLIGHTS_INDEX_NAME
        )

        pd_flights = self.pd_flights()

        assert_series_equal(pd_flights.dtypes, field_mappings.dtypes())

    def test_selected_fields(self):
        expected = ["timestamp", "DestWeather", "DistanceKilometers", "AvgTicketPrice"]

        field_mappings = FieldMappings(
            client=ES_TEST_CLIENT,
            index_pattern=FLIGHTS_INDEX_NAME,
            display_names=expected,
        )

        pd_flights = self.pd_flights()[expected]

        assert_series_equal(pd_flights.dtypes, field_mappings.dtypes())
