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
import elasticsearch
import pytest

import eland as ed
from eland.tests import ES_TEST_CLIENT
from eland.tests.common import TestData


class TestClientEq(TestData):

    def test_perform_request(self):
        client = ed.Client(ES_TEST_CLIENT)

        response = client.perform_request("GET", "/_cat/indices/flights")

        # yellow open flights TNUv0iysQSi7F-N5ykWfWQ 1 1 13059 0 5.7mb 5.7mb
        tokens = response.split(' ')

        assert tokens[2] == 'flights'
        assert tokens[6] == '13059'

    def test_bad_perform_request(self):
        client = ed.Client(ES_TEST_CLIENT)

        with pytest.raises(elasticsearch.exceptions.NotFoundError):
            response = client.perform_request("GET", "/_cat/indices/non_existant_index")
