# Licensed to Elasticsearch B.V under one or more agreements.
# Elasticsearch B.V licenses this file to you under the Apache 2.0 License.
# See the LICENSE file in the project root for more information

import numpy as np
import pandas as pd
from pandas.testing import assert_index_equal

# File called _pytest for PyCharm compatability
from eland.field_mappings import FieldMappings
from eland.tests import FLIGHTS_INDEX_NAME, ES_TEST_CLIENT
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
