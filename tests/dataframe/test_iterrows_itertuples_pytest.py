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
from pandas.testing import assert_series_equal

from tests.common import TestData


class TestDataFrameIterrowsItertuples(TestData):
    def test_iterrows(self):
        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()

        ed_flights_iterrows = ed_flights.iterrows()
        pd_flights_iterrows = pd_flights.iterrows()

        for ed_index, ed_row in ed_flights_iterrows:
            pd_index, pd_row = next(pd_flights_iterrows)

            assert ed_index == pd_index
            assert_series_equal(ed_row, pd_row)

        # Assert that both are the same length and are exhausted.
        with pytest.raises(StopIteration):
            next(ed_flights_iterrows)
        with pytest.raises(StopIteration):
            next(pd_flights_iterrows)

    def test_itertuples(self):
        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()

        ed_flights_itertuples = list(ed_flights.itertuples(name=None))
        pd_flights_itertuples = list(pd_flights.itertuples(name=None))

        def assert_tuples_almost_equal(left, right):
            # Shim which uses pytest.approx() for floating point values inside tuples.
            assert len(left) == len(right)
            assert all(
                (lt == rt)  # Not floats? Use ==
                if not isinstance(lt, float) and not isinstance(rt, float)
                else (lt == pytest.approx(rt))  # If both are floats use pytest.approx()
                for lt, rt in zip(left, right)
            )

        for ed_tuple, pd_tuple in zip(ed_flights_itertuples, pd_flights_itertuples):
            assert_tuples_almost_equal(ed_tuple, pd_tuple)
