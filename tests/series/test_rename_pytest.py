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
from tests import ES_TEST_CLIENT, FLIGHTS_INDEX_NAME
from tests.common import TestData, assert_pandas_eland_series_equal


class TestSeriesRename(TestData):
    def test_rename(self):
        pd_carrier = self.pd_flights()["Carrier"]
        ed_carrier = ed.Series(ES_TEST_CLIENT, FLIGHTS_INDEX_NAME, "Carrier")

        assert_pandas_eland_series_equal(pd_carrier, ed_carrier)

        pd_renamed = pd_carrier.rename("renamed")
        ed_renamed = ed_carrier.rename("renamed")

        print(pd_renamed)
        print(ed_renamed)

        print(ed_renamed.es_info())

        assert_pandas_eland_series_equal(pd_renamed, ed_renamed)

        pd_renamed2 = pd_renamed.rename("renamed2")
        ed_renamed2 = ed_renamed.rename("renamed2")

        print(ed_renamed2.es_info())

        assert "renamed2" == ed_renamed2.name

        assert_pandas_eland_series_equal(pd_renamed2, ed_renamed2)
