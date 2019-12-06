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

from eland.tests.common import TestData


class TestDataFrameShape(TestData):

    def test_ecommerce_shape(self):
        pd_ecommerce = self.pd_ecommerce()
        ed_ecommerce = self.ed_ecommerce()

        pd_shape = pd_ecommerce.shape
        ed_shape = ed_ecommerce.shape

        assert pd_shape == ed_shape

    def test_flights_shape(self):
        pd_flights = self.pd_flights()
        ed_flights = self.ed_flights()

        pd_shape = pd_flights.shape
        ed_shape = ed_flights.shape

        assert pd_shape == ed_shape
