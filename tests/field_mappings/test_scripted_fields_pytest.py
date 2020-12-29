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
from io import StringIO

import numpy as np

from eland.field_mappings import FieldMappings
from tests import ES_TEST_CLIENT, FLIGHTS_INDEX_NAME
from tests.common import TestData


class TestScriptedFields(TestData):
    def test_add_new_scripted_field(self):
        ed_field_mappings = FieldMappings(
            client=ES_TEST_CLIENT, index_pattern=FLIGHTS_INDEX_NAME
        )

        ed_field_mappings.add_scripted_field(
            "scripted_field_None", None, np.dtype("int64")
        )

        # note 'None' is printed as 'NaN' in index, but .index shows it is 'None'
        # buf = StringIO()
        # ed_field_mappings.info_es(buf)
        # print(buf.getvalue())

        expected = self.pd_flights().columns.to_list()
        expected.append(None)

        assert expected == ed_field_mappings.display_names

    def test_add_duplicate_scripted_field(self):
        ed_field_mappings = FieldMappings(
            client=ES_TEST_CLIENT, index_pattern=FLIGHTS_INDEX_NAME
        )

        ed_field_mappings.add_scripted_field(
            "scripted_field_Carrier", "Carrier", np.dtype("int64")
        )

        # note 'None' is printed as 'NaN' in index, but .index shows it is 'None'
        buf = StringIO()
        ed_field_mappings.es_info(buf)
        print(buf.getvalue())

        expected = self.pd_flights().columns.to_list()
        expected.remove("Carrier")
        expected.append("Carrier")

        assert expected == ed_field_mappings.display_names
