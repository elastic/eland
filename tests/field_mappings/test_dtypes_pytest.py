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
from pandas.testing import assert_series_equal

from eland.field_mappings import FieldMappings
from tests import ES_TEST_CLIENT, FLIGHTS_INDEX_NAME
from tests.common import TestData


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
