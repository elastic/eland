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

# File called _pytest for PyCharm compatibility
import eland as ed
from tests import ES_TEST_CLIENT, FLIGHTS_INDEX_NAME
from tests.common import TestData, assert_pandas_eland_series_equal


class TestSeriesSample(TestData):
    SEED = 42

    def build_from_index(self, ed_series):
        ed2pd_series = ed_series.to_pandas()
        return self.pd_flights()["Carrier"].iloc[ed2pd_series.index]

    def test_sample(self):
        ed_s = ed.Series(ES_TEST_CLIENT, FLIGHTS_INDEX_NAME, "Carrier")
        pd_s = self.build_from_index(ed_s.sample(n=10, random_state=self.SEED))

        ed_s_sample = ed_s.sample(n=10, random_state=self.SEED)
        assert_pandas_eland_series_equal(pd_s, ed_s_sample)
