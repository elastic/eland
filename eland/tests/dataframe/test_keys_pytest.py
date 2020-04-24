# Licensed to Elasticsearch B.V under one or more agreements.
# Elasticsearch B.V licenses this file to you under the Apache 2.0 License.
# See the LICENSE file in the project root for more information

# File called _pytest for PyCharm compatability

from pandas.testing import assert_index_equal

from eland.tests.common import TestData


class TestDataFrameKeys(TestData):
    def test_ecommerce_keys(self):
        pd_ecommerce = self.pd_ecommerce()
        ed_ecommerce = self.ed_ecommerce()

        pd_keys = pd_ecommerce.keys()
        ed_keys = ed_ecommerce.keys()

        assert_index_equal(pd_keys, ed_keys)

    def test_flights_keys(self):
        pd_flights = self.pd_flights()
        ed_flights = self.ed_flights()

        pd_keys = pd_flights.keys()
        ed_keys = ed_flights.keys()

        assert_index_equal(pd_keys, ed_keys)
