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
