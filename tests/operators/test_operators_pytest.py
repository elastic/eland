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

from eland.filter import (
    Equal,
    Greater,
    GreaterEqual,
    IsIn,
    IsNull,
    Less,
    LessEqual,
    Like,
    NotNull,
    Rlike,
    ScriptFilter,
    Startswith,
)


class TestOperators:
    def test_leaf_boolean_filter(self):
        assert GreaterEqual("a", 2).build() == {"range": {"a": {"gte": 2}}}
        assert LessEqual("a", 2).build() == {"range": {"a": {"lte": 2}}}
        assert Less("a", 2).build() == {"range": {"a": {"lt": 2}}}
        assert Equal("a", 2).build() == {"term": {"a": 2}}
        exp = Equal("a", 2)
        assert (~exp).build()["bool"], {"must_not": {"term": {"a": 2}}}
        assert Greater("a", 2).build() == {"range": {"a": {"gt": 2}}}
        assert IsIn("a", [1, 2, 3]).build() == {"terms": {"a": [1, 2, 3]}}
        assert Like("a", "a*b").build() == {"wildcard": {"a": "a*b"}}
        assert Rlike("a", "a*b").build() == {"regexp": {"a": "a*b"}}
        assert Startswith("a", "jj").build() == {"prefix": {"a": "jj"}}
        assert IsNull("a").build() == {"bool": {"must_not": {"exists": {"field": "a"}}}}
        assert NotNull("a").build() == {"exists": {"field": "a"}}
        assert ScriptFilter(
            'doc["num1"].value > params.param1', lang="painless", params={"param1": 5}
        ).build() == {
            "script": {
                "script": {
                    "lang": "painless",
                    "source": 'doc["num1"].value > params.param1',
                    "params": {"param1": 5},
                }
            }
        }
        assert IsIn("ids", [1, 2, 3]).build() == {"ids": {"values": [1, 2, 3]}}

    def test_and_filter1(self):
        exp = GreaterEqual("a", 2) & Less("b", 3)
        assert exp.build() == {
            "bool": {
                "must": [{"range": {"a": {"gte": 2}}}, {"range": {"b": {"lt": 3}}}]
            }
        }

    def test_and_filter2(self):
        exp = GreaterEqual("a", 2) & Less("b", 3) & Equal("c", 4)
        print(exp.build())
        assert exp.build() == {
            "bool": {
                "must": [
                    {"range": {"a": {"gte": 2}}},
                    {"range": {"b": {"lt": 3}}},
                    {"term": {"c": 4}},
                ]
            }
        }

    def test_and_filter3(self):
        exp = GreaterEqual("a", 2) & (Less("b", 3) & Equal("c", 4))
        assert exp.build() == {
            "bool": {
                "must": [
                    {"range": {"b": {"lt": 3}}},
                    {"term": {"c": 4}},
                    {"range": {"a": {"gte": 2}}},
                ]
            }
        }

    def test_or_filter1(self):
        exp = GreaterEqual("a", 2) | Less("b", 3)
        assert exp.build() == {
            "bool": {
                "should": [{"range": {"a": {"gte": 2}}}, {"range": {"b": {"lt": 3}}}]
            }
        }

    def test_or_filter2(self):
        exp = GreaterEqual("a", 2) | Less("b", 3) | Equal("c", 4)
        assert exp.build() == {
            "bool": {
                "should": [
                    {"range": {"a": {"gte": 2}}},
                    {"range": {"b": {"lt": 3}}},
                    {"term": {"c": 4}},
                ]
            }
        }

    def test_or_filter3(self):
        exp = GreaterEqual("a", 2) | (Less("b", 3) | Equal("c", 4))
        assert exp.build() == {
            "bool": {
                "should": [
                    {"range": {"b": {"lt": 3}}},
                    {"term": {"c": 4}},
                    {"range": {"a": {"gte": 2}}},
                ]
            }
        }

    def test_not_filter(self):
        exp = ~GreaterEqual("a", 2)
        assert exp.build() == {"bool": {"must_not": {"range": {"a": {"gte": 2}}}}}

    def test_not_not_filter(self):
        exp = ~~GreaterEqual("a", 2)

        assert exp.build() == {
            "bool": {"must_not": {"bool": {"must_not": {"range": {"a": {"gte": 2}}}}}}
        }

    def test_not_and_filter(self):
        exp = ~(GreaterEqual("a", 2) & Less("b", 3))
        assert exp.build() == {
            "bool": {
                "must_not": {
                    "bool": {
                        "must": [
                            {"range": {"a": {"gte": 2}}},
                            {"range": {"b": {"lt": 3}}},
                        ]
                    }
                }
            }
        }

    def test_and_or_filter(self):
        exp = GreaterEqual("a", 2) & (Less("b", 3) | Equal("c", 4))
        assert exp.build() == {
            "bool": {
                "must": [
                    {"range": {"a": {"gte": 2}}},
                    {
                        "bool": {
                            "should": [{"range": {"b": {"lt": 3}}}, {"term": {"c": 4}}]
                        }
                    },
                ]
            }
        }

    def test_and_not_or_filter(self):
        exp = GreaterEqual("a", 2) & ~(Less("b", 3) | Equal("c", 4))
        assert exp.build() == {
            "bool": {
                "must": [
                    {"range": {"a": {"gte": 2}}},
                    {
                        "bool": {
                            "must_not": {
                                "bool": {
                                    "should": [
                                        {"range": {"b": {"lt": 3}}},
                                        {"term": {"c": 4}},
                                    ]
                                }
                            }
                        }
                    },
                ]
            }
        }

    def test_must_and_must_not_filter(self):
        exp = (GreaterEqual("a", 2) & GreaterEqual("b", 2)) & ~(IsIn("ids", [1, 2, 3]))
        a = exp.build()
        b = {
            "bool": {
                "must": [
                    {"range": {"a": {"gte": 2}}},
                    {"range": {"b": {"gte": 2}}},
                    {"bool": {"must_not": {"ids": {"values": [1, 2, 3]}}}},
                ]
            }
        }
        assert a == b

    def test_must_not_and_must_filter(self):
        exp = ~(IsIn("ids", [1, 2, 3])) & (GreaterEqual("a", 2) & GreaterEqual("b", 2))
        a = exp.build()
        b = {
            "bool": {
                "must": [
                    {"range": {"a": {"gte": 2}}},
                    {"range": {"b": {"gte": 2}}},
                    {"bool": {"must_not": {"ids": {"values": [1, 2, 3]}}}},
                ]
            }
        }
        assert a == b

    def test_must_not_or_must_filter(self):
        exp = ~(IsIn("ids", [1, 2, 3])) | (GreaterEqual("a", 2) | GreaterEqual("b", 2))
        a = exp.build()
        b = {
            "bool": {
                "should": [
                    {"range": {"a": {"gte": 2}}},
                    {"range": {"b": {"gte": 2}}},
                    {"bool": {"must_not": {"ids": {"values": [1, 2, 3]}}}},
                ]
            }
        }
        assert a == b

    def test_must_or_must_not_filter(self):
        exp = (GreaterEqual("a", 2) | GreaterEqual("b", 2)) | ~(IsIn("ids", [1, 2, 3]))
        a = exp.build()
        b = {
            "bool": {
                "should": [
                    {"range": {"a": {"gte": 2}}},
                    {"range": {"b": {"gte": 2}}},
                    {"bool": {"must_not": {"ids": {"values": [1, 2, 3]}}}},
                ]
            }
        }
        assert a == b

    def test_complex_filter(self):
        exp = (
            Greater("a", 10)
            & ~Greater("a", 250)
            & (Equal("b", "X") | (Equal("b", "Y")))
            & (Equal("b", "Z"))
        )

        assert exp.build() == {
            "bool": {
                "must": [
                    {"range": {"a": {"gt": 10}}},
                    {"bool": {"must_not": {"range": {"a": {"gt": 250}}}}},
                    {"bool": {"should": [{"term": {"b": "X"}}, {"term": {"b": "Y"}}]}},
                    {"term": {"b": "Z"}},
                ]
            }
        }
