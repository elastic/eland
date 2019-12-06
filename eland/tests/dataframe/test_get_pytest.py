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


class TestDataFrameGet(TestData):

    def test_get_one_attribute(self):
        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()

        ed_get0 = ed_flights.get('Carrier')
        pd_get0 = pd_flights.get('Carrier')

        print(ed_get0, type(ed_get0))
        print(pd_get0, type(pd_get0))
