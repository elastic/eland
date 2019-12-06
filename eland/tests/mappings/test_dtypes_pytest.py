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

from pandas.util.testing import assert_series_equal

from eland.tests.common import TestData


class TestMappingsDtypes(TestData):

    def test_flights_dtypes_all(self):
        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()

        pd_dtypes = pd_flights.dtypes
        ed_dtypes = ed_flights._query_compiler._mappings.dtypes()

        assert_series_equal(pd_dtypes, ed_dtypes)

    def test_flights_dtypes_columns(self):
        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()[['Carrier', 'AvgTicketPrice', 'Cancelled']]

        pd_dtypes = pd_flights.dtypes
        ed_dtypes = ed_flights._query_compiler._mappings.dtypes(field_names=['Carrier', 'AvgTicketPrice', 'Cancelled'])

        assert_series_equal(pd_dtypes, ed_dtypes)
