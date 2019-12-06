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

from eland import Query
from eland.tests.common import TestData


class TestQueryCopy(TestData):

    def test_copy(self):
        q = Query()

        q.exists('field_a')
        q.exists('field_b', must=False)

        print(q.to_search_body())

        q1 = Query(q)

        q.exists('field_c', must=False)
        q1.exists('field_c1', must=False)

        print(q.to_search_body())
        print(q1.to_search_body())
