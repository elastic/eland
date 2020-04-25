# Licensed to Elasticsearch B.V under one or more agreements.
# Elasticsearch B.V licenses this file to you under the Apache 2.0 License.
# See the LICENSE file in the project root for more information

# File called _pytest for PyCharm compatability
import pandas as pd
from pandas.testing import assert_index_equal

from eland.tests.common import TestData


class TestGetFieldNames(TestData):
    def test_get_field_names_all(self):
        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()

        fields1 = ed_flights._query_compiler.get_field_names(
            include_scripted_fields=False
        )
        fields2 = ed_flights._query_compiler.get_field_names(
            include_scripted_fields=True
        )

        assert fields1 == fields2
        assert_index_equal(pd_flights.columns, pd.Index(fields1))

    def test_get_field_names_selected(self):
        ed_flights = self.ed_flights()[["Carrier", "AvgTicketPrice"]]
        pd_flights = self.pd_flights()[["Carrier", "AvgTicketPrice"]]

        fields1 = ed_flights._query_compiler.get_field_names(
            include_scripted_fields=False
        )
        fields2 = ed_flights._query_compiler.get_field_names(
            include_scripted_fields=True
        )

        assert fields1 == fields2
        assert_index_equal(pd_flights.columns, pd.Index(fields1))
