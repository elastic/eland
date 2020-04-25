# Licensed to Elasticsearch B.V under one or more agreements.
# Elasticsearch B.V licenses this file to you under the Apache 2.0 License.
# See the LICENSE file in the project root for more information

# File called _pytest for PyCharm compatability

from eland.field_mappings import FieldMappings
from eland.tests import ES_TEST_CLIENT, FLIGHTS_INDEX_NAME
from eland.tests.common import TestData


class TestRename(TestData):
    def test_single_rename(self):
        ed_field_mappings = FieldMappings(
            client=ES_TEST_CLIENT, index_pattern=FLIGHTS_INDEX_NAME
        )

        pd_flights_column_series = self.pd_flights().columns.to_series()

        assert (
            pd_flights_column_series.index.to_list() == ed_field_mappings.display_names
        )

        renames = {"DestWeather": "renamed_DestWeather"}

        # inplace rename
        ed_field_mappings.rename(renames)

        assert (
            pd_flights_column_series.rename(renames).index.to_list()
            == ed_field_mappings.display_names
        )

        get_renames = ed_field_mappings.get_renames()

        assert renames == get_renames

    def test_non_exists_rename(self):
        ed_field_mappings = FieldMappings(
            client=ES_TEST_CLIENT, index_pattern=FLIGHTS_INDEX_NAME
        )

        pd_flights_column_series = self.pd_flights().columns.to_series()

        assert (
            pd_flights_column_series.index.to_list() == ed_field_mappings.display_names
        )

        renames = {"unknown": "renamed_unknown"}

        # inplace rename - in this case it has no effect
        ed_field_mappings.rename(renames)

        assert (
            pd_flights_column_series.index.to_list() == ed_field_mappings.display_names
        )

        get_renames = ed_field_mappings.get_renames()

        assert not get_renames

    def test_exists_and_non_exists_rename(self):
        ed_field_mappings = FieldMappings(
            client=ES_TEST_CLIENT, index_pattern=FLIGHTS_INDEX_NAME
        )

        pd_flights_column_series = self.pd_flights().columns.to_series()

        assert (
            pd_flights_column_series.index.to_list() == ed_field_mappings.display_names
        )

        renames = {
            "unknown": "renamed_unknown",
            "DestWeather": "renamed_DestWeather",
            "unknown2": "renamed_unknown2",
            "Carrier": "renamed_Carrier",
        }

        # inplace rename - only real names get renamed
        ed_field_mappings.rename(renames)

        assert (
            pd_flights_column_series.rename(renames).index.to_list()
            == ed_field_mappings.display_names
        )

        get_renames = ed_field_mappings.get_renames()

        assert {
            "Carrier": "renamed_Carrier",
            "DestWeather": "renamed_DestWeather",
        } == get_renames

    def test_multi_rename(self):
        ed_field_mappings = FieldMappings(
            client=ES_TEST_CLIENT, index_pattern=FLIGHTS_INDEX_NAME
        )

        pd_flights_column_series = self.pd_flights().columns.to_series()

        assert (
            pd_flights_column_series.index.to_list() == ed_field_mappings.display_names
        )

        renames = {
            "DestWeather": "renamed_DestWeather",
            "renamed_DestWeather": "renamed_renamed_DestWeather",
        }

        # inplace rename - only first rename gets renamed
        ed_field_mappings.rename(renames)

        assert (
            pd_flights_column_series.rename(renames).index.to_list()
            == ed_field_mappings.display_names
        )

        get_renames = ed_field_mappings.get_renames()

        assert {"DestWeather": "renamed_DestWeather"} == get_renames
