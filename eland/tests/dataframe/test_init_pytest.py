# Licensed to Elasticsearch B.V under one or more agreements.
# Elasticsearch B.V licenses this file to you under the Apache 2.0 License.
# See the LICENSE file in the project root for more information

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
