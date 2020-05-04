# Licensed to Elasticsearch B.V under one or more agreements.
# Elasticsearch B.V licenses this file to you under the Apache 2.0 License.
# See the LICENSE file in the project root for more information

# File called _pytest for PyCharm compatibility
import eland as ed
from eland.tests import ES_TEST_CLIENT
from eland.tests import FLIGHTS_INDEX_NAME
from eland.tests.common import TestData
from eland.tests.common import assert_pandas_eland_series_equal


class TestSeriesSample(TestData):
    SEED = 42

    def build_from_index(self, ed_series):
        ed2pd_series = ed_series._to_pandas()
        return self.pd_flights()["Carrier"].iloc[ed2pd_series.index]

    def test_sample(self):
        ed_s = ed.Series(ES_TEST_CLIENT, FLIGHTS_INDEX_NAME, "Carrier")
        pd_s = self.build_from_index(ed_s.sample(n=10, random_state=self.SEED))

        ed_s_sample = ed_s.sample(n=10, random_state=self.SEED)
        assert_pandas_eland_series_equal(pd_s, ed_s_sample)
