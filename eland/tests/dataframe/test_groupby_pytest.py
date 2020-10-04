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

import pytest
from pandas.testing import assert_frame_equal
from eland.tests.common import TestData


class TestGroupbyDataFrame(TestData):
    funcs = ["max", "min", "mean", "sum"]
    extended_funcs = ["median", "mad", "var", "std"]
    filter_data = [
        "AvgTicketPrice",
        "Cancelled",
        "dayOfWeek",
        "timestamp",
        "DestCountry",
    ]

    @pytest.mark.parametrize("numeric_only", [True])
    def test_groupby_aggregate(self, numeric_only):
        # TODO numeric_only False and None
        # TODO Add more tests
        pd_flights = self.pd_flights().filter(self.filter_data)
        ed_flights = self.ed_flights().filter(self.filter_data)

        pd_groupby = pd_flights.groupby("Cancelled").agg(self.funcs, numeric_only)
        ed_groupby = ed_flights.groupby("Cancelled").agg(self.funcs, numeric_only)

        # checking only values because dtypes are checked in other tests
        assert_frame_equal(pd_groupby, ed_groupby, check_exact=False, check_dtype=False)

    def test_groupby_single_agg(self):
        # Write tests when grouped is implemented in eland.
        # Should write single agg tests
        pass
