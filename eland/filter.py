# Copyright 2020 Elasticsearch BV
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Originally based on code in MIT-licensed pandasticsearch filters

from typing import Dict, Any, List, Optional, Union, cast


class BooleanFilter:
    def __init__(self) -> None:
        self._filter: Dict[str, Any] = {}

    def __and__(self, x: "BooleanFilter") -> "BooleanFilter":
        # Combine results
        if isinstance(self, AndFilter):
            if "must_not" in x.subtree:
                # nest a must_not under a must
                self.subtree["must"].append(x.build())  # 'build includes bool'
            else:
                # append a must to a must
                self.subtree["must"].append(x.subtree)  # 'subtree strips bool'
            return self
        elif isinstance(x, AndFilter):
            if "must_not" in self.subtree:
                x.subtree["must"].append(self.build())
            else:
                x.subtree["must"].append(self.subtree)
            return x
        return AndFilter(self, x)

    def __or__(self, x: "BooleanFilter") -> "BooleanFilter":
        # Combine results
        if isinstance(self, OrFilter):
            if "must_not" in x.subtree:
                self.subtree["should"].append(x.build())
            else:
                self.subtree["should"].append(x.subtree)
            return self
        elif isinstance(x, OrFilter):
            if "must_not" in self.subtree:
                x.subtree["should"].append(self.build())
            else:
                x.subtree["should"].append(self.subtree)
            return x
        return OrFilter(self, x)

    def __invert__(self) -> "BooleanFilter":
        return NotFilter(self)

    def empty(self) -> bool:
        return not bool(self._filter)

    def __repr__(self) -> str:
        return str(self._filter)

    @property
    def subtree(self) -> Dict[str, Any]:
        if "bool" in self._filter:
            return cast(Dict[str, Any], self._filter["bool"])
        else:
            return self._filter

    def build(self) -> Dict[str, Any]:
        return self._filter


# Binary operator
class AndFilter(BooleanFilter):
    def __init__(self, *args: BooleanFilter) -> None:
        super().__init__()
        self._filter = {"bool": {"must": [x.build() for x in args]}}


class OrFilter(BooleanFilter):
    def __init__(self, *args: BooleanFilter) -> None:
        super().__init__()
        self._filter = {"bool": {"should": [x.build() for x in args]}}


class NotFilter(BooleanFilter):
    def __init__(self, x: BooleanFilter) -> None:
        super().__init__()
        self._filter = {"bool": {"must_not": x.build()}}


# LeafBooleanFilter
class GreaterEqual(BooleanFilter):
    def __init__(self, field: str, value: Any) -> None:
        super().__init__()
        self._filter = {"range": {field: {"gte": value}}}


class Greater(BooleanFilter):
    def __init__(self, field: str, value: Any) -> None:
        super().__init__()
        self._filter = {"range": {field: {"gt": value}}}


class LessEqual(BooleanFilter):
    def __init__(self, field: str, value: Any) -> None:
        super().__init__()
        self._filter = {"range": {field: {"lte": value}}}


class Less(BooleanFilter):
    def __init__(self, field: str, value: Any) -> None:
        super().__init__()
        self._filter = {"range": {field: {"lt": value}}}


class Equal(BooleanFilter):
    def __init__(self, field: str, value: Any) -> None:
        super().__init__()
        self._filter = {"term": {field: value}}


class IsIn(BooleanFilter):
    def __init__(self, field: str, value: List[Any]) -> None:
        super().__init__()
        if field == "ids":
            self._filter = {"ids": {"values": value}}
        else:
            self._filter = {"terms": {field: value}}


class Like(BooleanFilter):
    def __init__(self, field: str, value: str) -> None:
        super().__init__()
        self._filter = {"wildcard": {field: value}}


class Rlike(BooleanFilter):
    def __init__(self, field: str, value: str) -> None:
        super().__init__()
        self._filter = {"regexp": {field: value}}


class Startswith(BooleanFilter):
    def __init__(self, field: str, value: str) -> None:
        super().__init__()
        self._filter = {"prefix": {field: value}}


class IsNull(BooleanFilter):
    def __init__(self, field: str) -> None:
        super().__init__()
        self._filter = {"missing": {"field": field}}


class NotNull(BooleanFilter):
    def __init__(self, field: str) -> None:
        super().__init__()
        self._filter = {"exists": {"field": field}}


class ScriptFilter(BooleanFilter):
    def __init__(
        self,
        inline: str,
        lang: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        script: Dict[str, Union[str, Dict[str, Any]]] = {"source": inline}
        if lang is not None:
            script["lang"] = lang
        if params is not None:
            script["params"] = params
        self._filter = {"script": {"script": script}}


class QueryFilter(BooleanFilter):
    def __init__(self, query: Dict[str, Any]) -> None:
        super().__init__()
        self._filter = query
