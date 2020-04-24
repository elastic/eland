# Licensed to Elasticsearch B.V under one or more agreements.
# Elasticsearch B.V licenses this file to you under the Apache 2.0 License.
# See the LICENSE file in the project root for more information

# File called _pytest for PyCharm compatability
import eland as ed
from eland.tests import ES_TEST_CLIENT
from eland.tests import FLIGHTS_INDEX_NAME
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
