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

from pandas.testing import assert_index_equal, assert_series_equal

from tests.common import TestData


class TestDataFrameIterrowsItertuples(TestData):
    def test_iterrows(self):
        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()

        ed_flights_iterrows = ed_flights.iterrows()
        pd_flights_iterrows = pd_flights.iterrows()

        assert len(list(ed_flights_iterrows)) == len(list(pd_flights_iterrows))

        for ed_index, ed_row in ed_flights_iterrows:

            pd_index, pd_row = next(pd_flights_iterrows)

            assert_index_equal(ed_index, pd_index)
            assert_series_equal(ed_row, pd_row)

        for pd_index, pd_row in pd_flights_iterrows:

            ed_index, ed_row = next(ed_flights_iterrows)

            assert_index_equal(pd_index, ed_index)
            assert_series_equal(pd_row, ed_row)

    def test_itertuples(self):
        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()

        ed_flights_itertuples = ed_flights.itertuples(name=None)
        pd_flights_itertuples = pd_flights.itertuples(name=None)

        assert len(list(ed_flights_itertuples)) == len(list(pd_flights_itertuples))

        for ed_row in ed_flights_itertuples:

            pd_row = next(pd_flights_itertuples)

            assert ed_row == pd_row

        for pd_row in pd_flights_itertuples:

            ed_row = next(ed_flights_itertuples)

            assert pd_row == ed_row
