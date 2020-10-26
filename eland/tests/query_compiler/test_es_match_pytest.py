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

from eland.query_compiler import QueryCompiler
from eland.tests.common import TestData


class TestEsMatch(TestData):
    def test_es_match(self):
        df = self.ed_ecommerce()
        query_compiler: QueryCompiler = df._query_compiler

        filter = query_compiler.es_match(
            "joe", ["customer_full_name"], analyzer="my-analyzer", fuzziness="1..2"
        )
        assert filter.build() == {
            "match": {
                "customer_full_name": {
                    "query": "joe",
                    "analyzer": "my-analyzer",
                    "fuzziness": "1..2",
                }
            }
        }

        filter = query_compiler.es_match(
            "joe", ["customer_last_name", "customer_first_name"]
        )
        assert filter.build() == {
            "multi_match": {
                "query": "joe",
                "fields": ["customer_last_name", "customer_first_name"],
            }
        }

    def test_es_match_must_not_match(self):
        df = self.ed_ecommerce()

        # single match
        df2 = df.es_match("joe", columns=["customer_full_name"], must_not_match=True)
        query_params, _ = df2._query_compiler._operations._resolve_tasks(
            df2._query_compiler
        )
        assert query_params.query.to_search_body() == {
            "query": {
                "bool": {
                    "must_not": {"match": {"customer_full_name": {"query": "joe"}}}
                }
            }
        }

        # multi_match
        df2 = df.es_match(
            "joe",
            columns=["customer_first_name", "customer_last_name"],
            must_not_match=True,
        )
        query_params, _ = df2._query_compiler._operations._resolve_tasks(
            df2._query_compiler
        )
        assert query_params.query.to_search_body() == {
            "query": {
                "bool": {
                    "must_not": {
                        "multi_match": {
                            "fields": [
                                "customer_first_name",
                                "customer_last_name",
                            ],
                            "query": "joe",
                        }
                    }
                }
            }
        }

    def test_es_match_phrase(self):
        df = self.ed_ecommerce()
        query_compiler: QueryCompiler = df._query_compiler

        filter = query_compiler.es_match(
            "joe", ["customer_full_name"], match_phrase=True
        )
        assert filter.build() == {
            "match_phrase": {
                "customer_full_name": {
                    "query": "joe",
                }
            }
        }

        filter = query_compiler.es_match(
            "joe", ["customer_last_name", "customer_first_name"], match_phrase=True
        )
        assert filter.build() == {
            "multi_match": {
                "query": "joe",
                "type": "phrase",
                "fields": ["customer_last_name", "customer_first_name"],
            }
        }

    def test_es_match_phrase_not_allowed_with_multi_match_type(self):
        df = self.ed_ecommerce()
        query_compiler: QueryCompiler = df._query_compiler

        with pytest.raises(ValueError) as e:
            query_compiler.es_match(
                "joe",
                ["customer_first_name", "customer_last_name"],
                match_phrase=True,
                multi_match_type="best_fields",
            )
        assert str(e.value) == (
            "match_phrase=True and multi_match_type='best_fields' "
            "are not compatible. Must be multi_match_type='phrase'"
        )

        filter = query_compiler.es_match(
            "joe",
            ["customer_last_name", "customer_first_name"],
            match_phrase=True,
            multi_match_type="phrase",
        )
        assert filter.build() == {
            "multi_match": {
                "query": "joe",
                "type": "phrase",
                "fields": ["customer_last_name", "customer_first_name"],
            }
        }

    def test_es_match_non_text_fields(self):
        df = self.ed_ecommerce()
        query_compiler: QueryCompiler = df._query_compiler

        with pytest.raises(ValueError) as e:
            query_compiler.es_match(
                "joe",
                [
                    "customer_first_name",
                    "order_date",
                    "customer_last_name",
                    "currency",
                    "order_*",
                ],
            )
        assert str(e.value) == (
            "Attempting to run es_match() on non-text fields (order_date=date, "
            "currency=keyword) means that these fields may not be analyzed properly. "
            "Consider reindexing these fields as text or use 'match_only_text_es_dtypes=False' "
            "to use match anyways"
        )

        filter = query_compiler.es_match(
            "joe",
            [
                "customer_first_name",
                "order_date",
                "customer_last_name",
                "currency",
                "order_*",
            ],
            match_only_text_fields=False,
        )
        assert filter.build() == {
            "multi_match": {
                "query": "joe",
                "lenient": True,
                "fields": [
                    "customer_first_name",
                    "order_date",
                    "customer_last_name",
                    "currency",
                    "order_*",
                ],
            }
        }
