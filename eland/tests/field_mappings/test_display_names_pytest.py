# Licensed to Elasticsearch B.V under one or more agreements.
# Elasticsearch B.V licenses this file to you under the Apache 2.0 License.
# See the LICENSE file in the project root for more information

# File called _pytest for PyCharm compatability
import pytest

from eland.field_mappings import FieldMappings
from eland.tests import ES_TEST_CLIENT, FLIGHTS_INDEX_NAME
from eland.tests.common import TestData


class TestDisplayNames(TestData):
    def test_init_all_fields(self):
        field_mappings = FieldMappings(
            client=ES_TEST_CLIENT, index_pattern=FLIGHTS_INDEX_NAME
        )

        expected = self.pd_flights().columns.to_list()

        assert expected == field_mappings.display_names

    def test_init_selected_fields(self):
        expected = ["timestamp", "DestWeather", "DistanceKilometers", "AvgTicketPrice"]

        field_mappings = FieldMappings(
            client=ES_TEST_CLIENT,
            index_pattern=FLIGHTS_INDEX_NAME,
            display_names=expected,
        )

        assert expected == field_mappings.display_names

    def test_set_display_names(self):
        expected = [
            "Cancelled",
            "timestamp",
            "DestWeather",
            "DistanceKilometers",
            "AvgTicketPrice",
        ]

        field_mappings = FieldMappings(
            client=ES_TEST_CLIENT, index_pattern=FLIGHTS_INDEX_NAME
        )

        field_mappings.display_names = expected

        assert expected == field_mappings.display_names

        # now set again
        new_expected = ["AvgTicketPrice", "timestamp"]

        field_mappings.display_names = new_expected
        assert new_expected == field_mappings.display_names

    def test_not_found_display_names(self):
        not_found = [
            "Cancelled",
            "timestamp",
            "DestWeather",
            "unknown",
            "DistanceKilometers",
            "AvgTicketPrice",
        ]

        field_mappings = FieldMappings(
            client=ES_TEST_CLIENT, index_pattern=FLIGHTS_INDEX_NAME
        )

        with pytest.raises(KeyError):
            field_mappings.display_names = not_found

        expected = self.pd_flights().columns.to_list()

        assert expected == field_mappings.display_names

    def test_invalid_list_type_display_names(self):
        field_mappings = FieldMappings(
            client=ES_TEST_CLIENT, index_pattern=FLIGHTS_INDEX_NAME
        )

        # not a list like object
        with pytest.raises(ValueError):
            field_mappings.display_names = 12.0

        # tuple is list like
        field_mappings.display_names = ("Cancelled", "DestWeather")

        expected = ["Cancelled", "DestWeather"]

        assert expected == field_mappings.display_names
