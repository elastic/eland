# Copyright 2020 Elasticsearch BV
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# File called _pytest for PyCharm compatibility

from eland.tests.common import TestData
from eland.tests.common import assert_pandas_eland_frame_equal


class TestDataFrameSample(TestData):
    def test_sample_basic(self):
        ed_flights = self.ed_flights()
        sample_ed_flights = ed_flights.sample(n=10)._to_pandas()
        assert len(sample_ed_flights) == 10

    def test_sample_on_boolean_filter(self):
        ed_flights = self.ed_flights()
        columns = ["timestamp", "OriginAirportID", "DestAirportID", "FlightDelayMin"]
        shape = ed_flights[columns].sample(n=5)._to_pandas().shape
        assert (5, 4) == shape

    def test_sample_head(self):
        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()

        pd_head_5 = pd_flights.head(5)
        ed_head_5 = ed_flights.head(5).sample(5)
        assert_pandas_eland_frame_equal(pd_head_5, ed_head_5)

    def test_sample_frac_values(self):
        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()

        pd_head_5 = pd_flights.head(5)
        ed_head_5 = ed_flights.head(5).sample(frac=1)
        assert_pandas_eland_frame_equal(pd_head_5, ed_head_5)

    def test_sample_frac_is(self):
        frac = 0.1
        ed_flights = self.ed_flights()

        ed_flights_sample = ed_flights.sample(frac=frac)._to_pandas()
        size = len(ed_flights._to_pandas())
        assert len(ed_flights_sample) <= int(round(frac * size))
