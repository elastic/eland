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

from tests.common import TestData
from pandas.testing import assert_index_equal, assert_series_equal


class TestDataFrameIterrowsItertuples(TestData):
    def test_iterrows(self):
        ed_flights_iterrows = self.ed_flights().iterrows()
        pd_flights_iterrows = self.pd_flights().iterrows()
        assert len(ed_flights_iterrows) == len(pd_flights_iterrows)

        for i in len(ed_flights_iterrows):
            ed_index, ed_row = next(ed_flights_iterrows)
            pd_index, pd_row = next(pd_flights_iterrows)
            assert_index_equal(ed_index, pd_index)
            assert_series_equal(ed_row, pd_row)

    def test_itertuples(self):
        ed_flights_itertuples = self.ed_flights().itertuples()
        pd_flights_itertuples = self.pd_flights().itertuples()
        assert len(ed_flights_itertuples) == len(pd_flights_itertuples)

        for i in len(ed_flights_itertuples):
            ed_row = next(ed_flights_itertuples)
            pd_row = next(pd_flights_itertuples)
            assert ed_row == pd_row
