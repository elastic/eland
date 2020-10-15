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

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List, Tuple

from eland import SortOrder
from eland.actions import HeadAction, SortIndexAction, TailAction
from eland.arithmetics import ArithmeticSeries

if TYPE_CHECKING:
    from .actions import PostProcessingAction  # noqa: F401
    from .filter import BooleanFilter  # noqa: F401
    from .index import Index  # noqa: F401
    from .operations import QueryParams  # noqa: F401
    from .query_compiler import QueryCompiler  # noqa: F401

RESOLVED_TASK_TYPE = Tuple["QueryParams", List["PostProcessingAction"]]


class Task(ABC):
    """
    Abstract class for tasks

    Parameters
    ----------
        task_type: str
            The task type (e.g. head, tail etc.)
    """

    def __init__(self, task_type: str):
        self._task_type = task_type

    @property
    def type(self) -> str:
        return self._task_type

    @abstractmethod
    def resolve_task(
        self,
        query_params: "QueryParams",
        post_processing: List["PostProcessingAction"],
        query_compiler: "QueryCompiler",
    ) -> RESOLVED_TASK_TYPE:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass


class SizeTask(Task):
    def __init__(self, task_type: str, index: "Index", count: int):
        super().__init__(task_type)
        self._sort_field = index.sort_field
        self._count = min(len(index), count)

    @abstractmethod
    def size(self) -> int:
        # must override
        pass


class HeadTask(SizeTask):
    def __init__(self, index: "Index", count: int):
        super().__init__("head", index, count)

    def __repr__(self) -> str:
        return f"('{self._task_type}': ('sort_field': '{self._sort_field}', 'count': {self._count}))"

    def resolve_task(
        self,
        query_params: "QueryParams",
        post_processing: List["PostProcessingAction"],
        query_compiler: "QueryCompiler",
    ) -> RESOLVED_TASK_TYPE:
        # head - sort asc, size n
        # |12345-------------|
        query_sort_field = self._sort_field
        query_sort_order = SortOrder.ASC
        query_size = self._count

        # If we are already postprocessing the query results, we just get 'head' of these
        # (note, currently we just append another head, we don't optimise by
        # overwriting previous head)
        if len(post_processing) > 0:
            post_processing.append(HeadAction(self._count))
            return query_params, post_processing

        if query_params.sort_field is None:
            query_params.sort_field = query_sort_field
        # if it is already sorted we use existing field

        if query_params.sort_order is None:
            query_params.sort_order = query_sort_order
        # if it is already sorted we get head of existing order

        if query_params.size is None:
            query_params.size = query_size
        else:
            # truncate if head is smaller
            if query_size < query_params.size:
                query_params.size = query_size

        return query_params, post_processing

    def size(self) -> int:
        return self._count


class TailTask(SizeTask):
    def __init__(self, index: "Index", count: int):
        super().__init__("tail", index, count)

    def resolve_task(
        self,
        query_params: "QueryParams",
        post_processing: List["PostProcessingAction"],
        query_compiler: "QueryCompiler",
    ) -> RESOLVED_TASK_TYPE:
        # tail - sort desc, size n, post-process sort asc
        # |-------------12345|
        query_sort_field = self._sort_field
        query_sort_order = SortOrder.DESC
        query_size = self._count

        # If this is a tail of a tail adjust settings and return
        if (
            query_params.size is not None
            and query_params.sort_order == query_sort_order
            and (
                len(post_processing) == 1
                and isinstance(post_processing[0], SortIndexAction)
            )
        ):
            if query_size < query_params.size:
                query_params.size = query_size
            return query_params, post_processing

        # If we are already postprocessing the query results, just get 'tail' of these
        # (note, currently we just append another tail, we don't optimise by
        # overwriting previous tail)
        if len(post_processing) > 0:
            post_processing.append(TailAction(self._count))
            return query_params, post_processing

        # If results are already constrained, just get 'tail' of these
        # (note, currently we just append another tail, we don't optimise by
        # overwriting previous tail)
        if query_params.size is not None:
            post_processing.append(TailAction(self._count))
            return query_params, post_processing
        else:
            query_params.size = query_size
        if query_params.sort_field is None:
            query_params.sort_field = query_sort_field
        if query_params.sort_order is None:
            query_params.sort_order = query_sort_order
        else:
            # reverse sort order
            query_params.sort_order = SortOrder.reverse(query_sort_order)

        post_processing.append(SortIndexAction())

        return query_params, post_processing

    def size(self) -> int:
        return self._count

    def __repr__(self) -> str:
        return f"('{self._task_type}': ('sort_field': '{self._sort_field}', 'count': {self._count}))"


class SampleTask(SizeTask):
    def __init__(self, index: "Index", count: int, random_state: int):
        super().__init__("sample", index, count)
        self._random_state = random_state

    def resolve_task(
        self,
        query_params: "QueryParams",
        post_processing: List["PostProcessingAction"],
        query_compiler: "QueryCompiler",
    ) -> RESOLVED_TASK_TYPE:
        query_params.query.random_score(self._random_state)

        query_sort_field = self._sort_field
        query_size = self._count

        if query_params.size is not None:
            query_params.size = min(query_size, query_params.size)
        else:
            query_params.size = query_size

        if query_params.sort_field is None:
            query_params.sort_field = query_sort_field

        post_processing.append(SortIndexAction())

        return query_params, post_processing

    def size(self) -> int:
        return self._count

    def __repr__(self) -> str:
        return f"('{self._task_type}': ('count': {self._count}))"


class QueryIdsTask(Task):
    def __init__(self, must: bool, ids: List[str], sort_index_by_ids: bool = False):
        """
        Parameters
        ----------
        must: bool
            Include or exclude these ids (must/must_not)

        ids: list
            ids for the filter
        """
        super().__init__("query_ids")

        self._must = must
        self._ids = ids
        self._sort_index_by_ids = sort_index_by_ids

    def resolve_task(
        self,
        query_params: "QueryParams",
        post_processing: List["PostProcessingAction"],
        query_compiler: "QueryCompiler",
    ) -> RESOLVED_TASK_TYPE:
        query_params.query.ids(self._ids, must=self._must)
        if self._sort_index_by_ids:
            post_processing.append(SortIndexAction(items=self._ids))
        return query_params, post_processing

    def __repr__(self) -> str:
        return f"('{self._task_type}': ('must': {self._must}, 'ids': {self._ids}))"


class QueryTermsTask(Task):
    def __init__(self, must: bool, field: str, terms: List[Any]):
        """
        Parameters
        ----------
        must: bool
            Include or exclude these ids (must/must_not)

        field: str
            field_name to filter

        terms: list
            field_values for filter
        """
        super().__init__("query_terms")

        self._must = must
        self._field = field
        self._terms = terms

    def resolve_task(
        self,
        query_params: "QueryParams",
        post_processing: List["PostProcessingAction"],
        query_compiler: "QueryCompiler",
    ) -> RESOLVED_TASK_TYPE:
        query_params.query.terms(self._field, self._terms, must=self._must)
        return query_params, post_processing

    def __repr__(self) -> str:
        return (
            f"('{self._task_type}': ('must': {self._must}, "
            f"'field': '{self._field}', 'terms': {self._terms}))"
        )


class QueryRegexpTask(Task):
    def __init__(self, field: str, value: str):
        """
        Parameters
        ----------
        field: str
            field_name to filter

        value: str
            regular expression pattern for filter
        """
        super().__init__("regexp")

        self._field = field
        self._value = value

    def resolve_task(
        self,
        query_params: "QueryParams",
        post_processing: List["PostProcessingAction"],
        query_compiler: "QueryCompiler",
    ) -> RESOLVED_TASK_TYPE:
        query_params.query.regexp(self._field, self._value)
        return query_params, post_processing

    def __repr__(self) -> str:
        return (
            f"('{self._task_type}': ('field': '{self._field}', 'value': {self._value}))"
        )


class BooleanFilterTask(Task):
    def __init__(self, boolean_filter: "BooleanFilter"):
        """
        Parameters
        ----------
        boolean_filter: BooleanFilter or str
            The filter to apply
        """
        super().__init__("boolean_filter")

        self._boolean_filter = boolean_filter

    def resolve_task(
        self,
        query_params: "QueryParams",
        post_processing: List["PostProcessingAction"],
        query_compiler: "QueryCompiler",
    ) -> RESOLVED_TASK_TYPE:
        query_params.query.update_boolean_filter(self._boolean_filter)
        return query_params, post_processing

    def __repr__(self) -> str:
        return f"('{self._task_type}': ('boolean_filter': {self._boolean_filter!r}))"


class ArithmeticOpFieldsTask(Task):
    def __init__(self, display_name: str, arithmetic_series: ArithmeticSeries):
        super().__init__("arithmetic_op_fields")

        self._display_name = display_name

        if not isinstance(arithmetic_series, ArithmeticSeries):
            raise TypeError(f"Expecting ArithmeticSeries got {type(arithmetic_series)}")
        self._arithmetic_series = arithmetic_series

    def update(self, display_name: str, arithmetic_series: ArithmeticSeries) -> None:
        self._display_name = display_name
        self._arithmetic_series = arithmetic_series

    def resolve_task(
        self,
        query_params: "QueryParams",
        post_processing: List["PostProcessingAction"],
        query_compiler: "QueryCompiler",
    ) -> RESOLVED_TASK_TYPE:
        # https://www.elastic.co/guide/en/elasticsearch/painless/current/painless-api-reference-shared-java-lang.html#painless-api-reference-shared-Math
        """
        "script_fields": {
            "field_name": {
            "script": {
                "source": "doc[self._left_field].value / self._right_field"
            }
            }
        }
        """
        if query_params.script_fields is None:
            query_params.script_fields = {}

        # TODO: Remove this once 'query_params' becomes a dataclass.
        assert isinstance(query_params.script_fields, dict)

        if self._display_name in query_params.script_fields:
            raise NotImplementedError(
                f"TODO code path - combine multiple ops "
                f"'{self}'\n{query_params.script_fields}\n"
                f"{self._display_name}\n{self._arithmetic_series.resolve()}"
            )

        query_params.script_fields[self._display_name] = {
            "script": {"source": self._arithmetic_series.resolve()}
        }

        return query_params, post_processing

    def __repr__(self) -> str:
        return (
            f"('{self._task_type}': ("
            f"'display_name': {self._display_name}, "
            f"'arithmetic_object': {self._arithmetic_series}"
            f"))"
        )
