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
from elasticsearch import Elasticsearch

import eland as ed
from eland.tests.common import TestData


class TestClientEq(TestData):

    def test_self_eq(self):
        es = Elasticsearch('localhost')

        client = ed.Client(es)

        assert client != es

        assert client == client

    def test_non_self_ne(self):
        es1 = Elasticsearch('localhost')
        es2 = Elasticsearch('localhost')

        client1 = ed.Client(es1)
        client2 = ed.Client(es2)

        assert client1 != client2
