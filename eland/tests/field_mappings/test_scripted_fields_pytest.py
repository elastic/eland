# Licensed to Elasticsearch B.V under one or more agreements.
# Elasticsearch B.V licenses this file to you under the Apache 2.0 License.
# See the LICENSE file in the project root for more information

# File called _pytest for PyCharm compatability
from io import StringIO

import numpy as np

from eland.field_mappings import FieldMappings
from eland.tests import FLIGHTS_INDEX_NAME, ES_TEST_CLIENT
from eland.tests.common import TestData


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
        ed_field_mappings.info_es(buf)
        print(buf.getvalue())

        expected = self.pd_flights().columns.to_list()
        expected.remove("Carrier")
        expected.append("Carrier")

        assert expected == ed_field_mappings.display_names
