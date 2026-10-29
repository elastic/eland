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

from unittest.mock import Mock

import pandas as pd
from pandas.testing import assert_index_equal

from eland.index import Index
from tests.common import TestData


class TestIndexToPandas(TestData):
    def test_to_pandas(self):
        assert_index_equal(
            self.pd_flights().index,
            self.ed_flights().index.to_pandas(),
        )

    def test_show_progress_is_forwarded(self):
        expected = pd.Index(["1", "2"])
        query_compiler = Mock()
        query_compiler.to_pandas.return_value = pd.DataFrame(index=expected)

        result = Index(query_compiler).to_pandas(show_progress=True)

        assert_index_equal(expected, result)
        query_compiler.to_pandas.assert_called_once_with(show_progress=True)
