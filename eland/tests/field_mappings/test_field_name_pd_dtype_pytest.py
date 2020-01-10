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
import pytest
from pandas.util.testing import assert_series_equal

import eland as ed
from eland.tests import FLIGHTS_INDEX_NAME, FLIGHTS_MAPPING
from eland.tests.common import ES_TEST_CLIENT
from eland.tests.common import TestData


class TestFieldNamePDDType(TestData):

    def test_all_formats(self):
        ed_field_mappings = ed.FieldMappings(
            client=ed.Client(ES_TEST_CLIENT),
            index_pattern=FLIGHTS_INDEX_NAME
        )

        pd_flights = self.pd_flights()

        assert_series_equal(pd_flights.dtypes, ed_field_mappings.dtypes())

        for es_field_name in FLIGHTS_MAPPING['mappings']['properties'].keys():
            pd_dtype = ed_field_mappings.field_name_pd_dtype(es_field_name)

            assert pd_flights[es_field_name].dtype == pd_dtype

    def test_non_existant(self):
        ed_field_mappings = ed.FieldMappings(
            client=ed.Client(ES_TEST_CLIENT),
            index_pattern=FLIGHTS_INDEX_NAME
        )

        with pytest.raises(KeyError):
            pd_dtype = ed_field_mappings.field_name_pd_dtype('unknown')
