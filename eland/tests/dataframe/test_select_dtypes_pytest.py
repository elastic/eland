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
import numpy as np

from eland.tests.common import TestData
from eland.tests.common import (
    assert_pandas_eland_frame_equal
)


class TestDataFrameSelectDTypes(TestData):

    def test_select_dtypes_include_number(self):
        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()

        ed_flights_numeric = ed_flights.select_dtypes(include=[np.number])
        pd_flights_numeric = pd_flights.select_dtypes(include=[np.number])

        assert_pandas_eland_frame_equal(pd_flights_numeric.head(103), ed_flights_numeric.head(103))

    def test_select_dtypes_exclude_number(self):
        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()

        ed_flights_non_numeric = ed_flights.select_dtypes(exclude=[np.number])
        pd_flights_non_numeric = pd_flights.select_dtypes(exclude=[np.number])

        assert_pandas_eland_frame_equal(pd_flights_non_numeric.head(103), ed_flights_non_numeric.head(103))
