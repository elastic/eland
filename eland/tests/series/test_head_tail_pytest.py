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
import eland as ed
from eland.tests import ES_TEST_CLIENT, FLIGHTS_INDEX_NAME
from eland.tests.common import TestData, assert_pandas_eland_series_equal


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
