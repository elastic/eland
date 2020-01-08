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
from pandas.util.testing import assert_series_equal

from eland.tests.common import TestData
from eland.tests.common import assert_pandas_eland_frame_equal


class TestDataFrameDtypes(TestData):

    def test_flights_dtypes(self):
        pd_flights = self.pd_flights()
        ed_flights = self.ed_flights()

        print(pd_flights.dtypes)
        print(ed_flights.dtypes)

        assert_series_equal(pd_flights.dtypes, ed_flights.dtypes)

        for i in range(0, len(pd_flights.dtypes) - 1):
            assert isinstance(pd_flights.dtypes[i], type(ed_flights.dtypes[i]))

    def test_flights_select_dtypes(self):
        pd_flights = self.pd_flights_small()
        ed_flights = self.ed_flights_small()

        assert_pandas_eland_frame_equal(
            pd_flights.select_dtypes(include=np.number),
            ed_flights.select_dtypes(include=np.number)
        )
