# Licensed to Elasticsearch B.V under one or more agreements.
# Elasticsearch B.V licenses this file to you under the Apache 2.0 License.
# See the LICENSE file in the project root for more information

# Originally based on code in MIT-licensed pandasticsearch filters

from typing import Dict, Any, List, Optional, Union, cast


class BooleanFilter:
    def __init__(self) -> None:
        self._filter: Dict[str, Any] = {}

    def __and__(self, x: "BooleanFilter") -> "BooleanFilter":
        if tuple(self.subtree.keys()) == ("must",):
            if "bool" in self._filter:
                self.subtree["must"].append(x.build())
            else:
                self.subtree["must"].append(x.subtree)
            return self
        elif tuple(x.subtree.keys()) == ("must",):
            if "bool" in x._filter:
                x.subtree["must"].append(self.build())
            else:
                x.subtree["must"].append(self.subtree)
            return x
        return AndFilter(self, x)

    def __or__(self, x: "BooleanFilter") -> "BooleanFilter":
        if tuple(self.subtree.keys()) == ("should",):
            if "bool" in x._filter:
                self.subtree["should"].append(x.build())
            else:
                self.subtree["should"].append(x.subtree)
            return self
        elif tuple(x.subtree.keys()) == ("should",):
            if "bool" in self._filter:
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
        return str(self.build())

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
