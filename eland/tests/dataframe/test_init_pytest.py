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

import pytest

import eland as ed
from eland.query_compiler import QueryCompiler
from eland.tests import ES_TEST_CLIENT
from eland.tests import FLIGHTS_INDEX_NAME


class TestDataFrameInit:
    def test_init(self):
        # Construct empty DataFrame (throws)
        with pytest.raises(ValueError):
            ed.DataFrame()

        # Construct invalid DataFrame (throws)
        with pytest.raises(ValueError):
            ed.DataFrame(es_client=ES_TEST_CLIENT)

        # Construct invalid DataFrame (throws)
        with pytest.raises(ValueError):
            ed.DataFrame(es_index_pattern=FLIGHTS_INDEX_NAME)

        # Good constructors
        ed.DataFrame(ES_TEST_CLIENT, FLIGHTS_INDEX_NAME)
        ed.DataFrame(es_client=ES_TEST_CLIENT, es_index_pattern=FLIGHTS_INDEX_NAME)

        qc = QueryCompiler(client=ES_TEST_CLIENT, index_pattern=FLIGHTS_INDEX_NAME)
        ed.DataFrame(_query_compiler=qc)
