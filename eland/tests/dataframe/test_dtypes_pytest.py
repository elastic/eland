# Licensed to Elasticsearch B.V under one or more agreements.
# Elasticsearch B.V licenses this file to you under the Apache 2.0 License.
# See the LICENSE file in the project root for more information

# File called _pytest for PyCharm compatability

import numpy as np
from pandas.testing import assert_series_equal

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
            ed_flights.select_dtypes(include=np.number),
        )
