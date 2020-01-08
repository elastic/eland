#  Copyright 2019 Elasticsearch BV
#
#      Licensed under the Apache License, Version 2.0 (the "License");
#      you may not use this file except in compliance with the License.
#      You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#      Unless required by applicable law or agreed to in writing, software
#      distributed under the License is distributed on an "AS IS" BASIS,
#      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#      See the License for the specific language governing permissions and
#      limitations under the License.

# File called _pytest for PyCharm compatability

import eland as ed
from eland.tests import ES_TEST_CLIENT, FLIGHTS_INDEX_NAME
from eland.tests.common import TestData


class TestRename(TestData):

    def test_single_rename(self):
        ed_field_mappings = ed.FieldMappings(
            client=ed.Client(ES_TEST_CLIENT),
            index_pattern=FLIGHTS_INDEX_NAME
        )

        pd_flights_column_series = self.pd_flights().columns.to_series()

        assert pd_flights_column_series.index.to_list() == ed_field_mappings.display_names

        renames = {'DestWeather': 'renamed_DestWeather'}

        # inplace rename
        ed_field_mappings.rename(renames)

        assert pd_flights_column_series.rename(renames).index.to_list() == ed_field_mappings.display_names

        get_renames = ed_field_mappings.get_renames()

        assert renames == get_renames

    def test_non_exists_rename(self):
        ed_field_mappings = ed.FieldMappings(
            client=ed.Client(ES_TEST_CLIENT),
            index_pattern=FLIGHTS_INDEX_NAME
        )

        pd_flights_column_series = self.pd_flights().columns.to_series()

        assert pd_flights_column_series.index.to_list() == ed_field_mappings.display_names

        renames = {'unknown': 'renamed_unknown'}

        # inplace rename - in this case it has no effect
        ed_field_mappings.rename(renames)

        assert pd_flights_column_series.index.to_list() == ed_field_mappings.display_names

        get_renames = ed_field_mappings.get_renames()

        assert not get_renames

    def test_exists_and_non_exists_rename(self):
        ed_field_mappings = ed.FieldMappings(
            client=ed.Client(ES_TEST_CLIENT),
            index_pattern=FLIGHTS_INDEX_NAME
        )

        pd_flights_column_series = self.pd_flights().columns.to_series()

        assert pd_flights_column_series.index.to_list() == ed_field_mappings.display_names

        renames = {'unknown': 'renamed_unknown', 'DestWeather': 'renamed_DestWeather', 'unknown2': 'renamed_unknown2',
                   'Carrier': 'renamed_Carrier'}

        # inplace rename - only real names get renamed
        ed_field_mappings.rename(renames)

        assert pd_flights_column_series.rename(renames).index.to_list() == ed_field_mappings.display_names

        get_renames = ed_field_mappings.get_renames()

        assert {'Carrier': 'renamed_Carrier', 'DestWeather': 'renamed_DestWeather'} == get_renames

    def test_multi_rename(self):
        ed_field_mappings = ed.FieldMappings(
            client=ed.Client(ES_TEST_CLIENT),
            index_pattern=FLIGHTS_INDEX_NAME
        )

        pd_flights_column_series = self.pd_flights().columns.to_series()

        assert pd_flights_column_series.index.to_list() == ed_field_mappings.display_names

        renames = {'DestWeather': 'renamed_DestWeather', 'renamed_DestWeather': 'renamed_renamed_DestWeather'}

        # inplace rename - only first rename gets renamed
        ed_field_mappings.rename(renames)

        assert pd_flights_column_series.rename(renames).index.to_list() == ed_field_mappings.display_names

        get_renames = ed_field_mappings.get_renames()

        assert {'DestWeather': 'renamed_DestWeather'} == get_renames
