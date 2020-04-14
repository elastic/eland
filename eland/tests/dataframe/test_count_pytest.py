# Copyright 2020 Elasticsearch BV
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
