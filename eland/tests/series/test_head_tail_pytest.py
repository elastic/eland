# Licensed to Elasticsearch B.V under one or more agreements.
# Elasticsearch B.V licenses this file to you under the Apache 2.0 License.
# See the LICENSE file in the project root for more information

# File called _pytest for PyCharm compatability
import eland as ed
from eland.tests import ES_TEST_CLIENT
from eland.tests import FLIGHTS_INDEX_NAME
from eland.tests.common import TestData
from eland.tests.common import assert_pandas_eland_series_equal


class TestSeriesHeadTail(TestData):
    def test_head_tail(self):
        pd_s = self.pd_flights()["Carrier"]
        ed_s = ed.Series(ES_TEST_CLIENT, FLIGHTS_INDEX_NAME, "Carrier")

        pd_s_head = pd_s.head(10)
        ed_s_head = ed_s.head(10)

        assert_pandas_eland_series_equal(pd_s_head, ed_s_head)

        pd_s_tail = pd_s.tail(10)
        ed_s_tail = ed_s.tail(10)

        assert_pandas_eland_series_equal(pd_s_tail, ed_s_tail)
