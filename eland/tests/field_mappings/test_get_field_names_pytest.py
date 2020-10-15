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

import numpy as np
import pandas as pd
from pandas.testing import assert_index_equal

# File called _pytest for PyCharm compatability
from eland.field_mappings import FieldMappings
from eland.tests import ES_TEST_CLIENT, FLIGHTS_INDEX_NAME
from eland.tests.common import TestData


class TestGetFieldNames(TestData):
    def test_get_field_names_all(self):
        ed_field_mappings = FieldMappings(
            client=ES_TEST_CLIENT, index_pattern=FLIGHTS_INDEX_NAME
        )
        pd_flights = self.pd_flights()

        fields1 = ed_field_mappings.get_field_names(include_scripted_fields=False)
        fields2 = ed_field_mappings.get_field_names(include_scripted_fields=True)

        assert fields1 == fields2
        assert_index_equal(pd_flights.columns, pd.Index(fields1))

    def test_get_field_names_selected(self):
        expected = ["Carrier", "AvgTicketPrice"]
        ed_field_mappings = FieldMappings(
            client=ES_TEST_CLIENT,
            index_pattern=FLIGHTS_INDEX_NAME,
            display_names=expected,
        )
        pd_flights = self.pd_flights()[expected]

        fields1 = ed_field_mappings.get_field_names(include_scripted_fields=False)
        fields2 = ed_field_mappings.get_field_names(include_scripted_fields=True)

        assert fields1 == fields2
        assert_index_equal(pd_flights.columns, pd.Index(fields1))

    def test_get_field_names_scripted(self):
        expected = ["Carrier", "AvgTicketPrice"]
        ed_field_mappings = FieldMappings(
            client=ES_TEST_CLIENT,
            index_pattern=FLIGHTS_INDEX_NAME,
            display_names=expected,
        )
        pd_flights = self.pd_flights()[expected]

        fields1 = ed_field_mappings.get_field_names(include_scripted_fields=False)
        fields2 = ed_field_mappings.get_field_names(include_scripted_fields=True)

        assert fields1 == fields2
        assert_index_equal(pd_flights.columns, pd.Index(fields1))

        # now add scripted field
        ed_field_mappings.add_scripted_field(
            "scripted_field_None", None, np.dtype("int64")
        )

        fields3 = ed_field_mappings.get_field_names(include_scripted_fields=False)
        fields4 = ed_field_mappings.get_field_names(include_scripted_fields=True)

        assert fields1 == fields3
        fields1.append("scripted_field_None")
        assert fields1 == fields4
