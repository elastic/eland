# Licensed to Elasticsearch B.V under one or more agreements.
# Elasticsearch B.V licenses this file to you under the Apache 2.0 License.
# See the LICENSE file in the project root for more information

# File called _pytest for PyCharm compatability

from pandas.testing import assert_series_equal

from eland.tests.common import TestData


class TestDataFrameCount(TestData):
    def test_ecommerce_count(self):
        pd_ecommerce = self.pd_ecommerce()
        ed_ecommerce = self.ed_ecommerce()

        pd_count = pd_ecommerce.count()
        ed_count = ed_ecommerce.count()

        print(pd_count)
        print(ed_count)

        assert_series_equal(pd_count, ed_count)
