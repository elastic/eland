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
import pandas as pd

import eland as ed
from eland.tests import ES_TEST_CLIENT, FLIGHTS_INDEX_NAME
from eland.tests.common import TestData


class TestSeriesRepr(TestData):
    def test_repr_flights_carrier(self):
        pd_s = self.pd_flights()["Carrier"]
        ed_s = ed.Series(ES_TEST_CLIENT, FLIGHTS_INDEX_NAME, "Carrier")

        pd_repr = repr(pd_s)
        ed_repr = repr(ed_s)

        assert pd_repr == ed_repr

    def test_repr_flights_carrier_5(self):
        pd_s = self.pd_flights()["Carrier"].head(5)
        ed_s = ed.Series(ES_TEST_CLIENT, FLIGHTS_INDEX_NAME, "Carrier").head(5)

        pd_repr = repr(pd_s)
        ed_repr = repr(ed_s)

        assert pd_repr == ed_repr

    def test_repr_empty_series(self):
        pd_s = self.pd_flights()["Carrier"].head(0)
        ed_s = ed.Series(ES_TEST_CLIENT, FLIGHTS_INDEX_NAME, "Carrier").head(0)
        assert repr(pd_s) == repr(ed_s)

    def test_series_repr_pd_get_option_none(self):
        show_dimensions = pd.get_option("display.show_dimensions")
        show_rows = pd.get_option("display.max_rows")
        try:
            pd.set_option("display.show_dimensions", False)
            pd.set_option("display.max_rows", None)

            ed_flights = self.ed_flights()["Cancelled"].head(40).__repr__()
            pd_flights = self.pd_flights()["Cancelled"].head(40).__repr__()

            assert ed_flights == pd_flights
        finally:
            pd.set_option("display.max_rows", show_rows)
            pd.set_option("display.show_dimensions", show_dimensions)
