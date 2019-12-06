#  Copyright 2019 Elasticsearch BV
#
#      Licensed under the Apache License, Version 2.0 (the "License");
#      you may not use this file except in compliance with the License.
#      You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#      Unless required by applicable law or agreed to in writing, software
#      distributed under the License is distributed on an "AS IS" BASIS,
#      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#      See the License for the specific language governing permissions and
#      limitations under the License.

# File called _pytest for PyCharm compatability
import eland as ed
from eland.tests import ES_TEST_CLIENT
from eland.tests import FLIGHTS_INDEX_NAME
from eland.tests.common import TestData
from eland.tests.common import assert_pandas_eland_series_equal


class TestSeriesName(TestData):

    def test_name(self):
        # deep copy pandas DataFrame as .name alters this reference frame
        pd_series = self.pd_flights()['Carrier'].copy(deep=True)
        ed_series = ed.Series(ES_TEST_CLIENT, FLIGHTS_INDEX_NAME, 'Carrier')

        assert_pandas_eland_series_equal(pd_series, ed_series)
        assert ed_series.name == pd_series.name

        pd_series.name = "renamed1"
        ed_series.name = "renamed1"

        assert_pandas_eland_series_equal(pd_series, ed_series)
        assert ed_series.name == pd_series.name

        pd_series.name = "renamed2"
        ed_series.name = "renamed2"

        assert_pandas_eland_series_equal(pd_series, ed_series)
        assert ed_series.name == pd_series.name
