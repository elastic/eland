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
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from eland import SortOrder


if TYPE_CHECKING:
    import pandas as pd  # type: ignore


class PostProcessingAction(ABC):
    def __init__(self, action_type: str) -> None:
        """
        Abstract class for postprocessing actions

        Parameters
        ----------
            action_type: str
                The action type (e.g. sort_index, head etc.)
        """
        self._action_type = action_type

    @property
    def type(self) -> str:
        return self._action_type

    @abstractmethod
    def resolve_action(self, df: "pd.DataFrame") -> "pd.DataFrame":
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass


class SortIndexAction(PostProcessingAction):
    def __init__(self, sort_orders: Optional[List[SortOrder]] = None) -> None:
        super().__init__("sort_index")
        self._sort_orders = sort_orders

    def resolve_action(self, df: "pd.DataFrame") -> "pd.DataFrame":
        if self._sort_orders is None:
            return df.sort_index()
        for i, sort_order in list(enumerate(self._sort_orders))[::-1]:
            df = df.sort_index(level=i, ascending=sort_order == SortOrder.ASC)
        return df

    def __repr__(self) -> str:
        return f"('{self.type}': ('sort_orders': {self._sort_orders})"


class HeadAction(PostProcessingAction):
    def __init__(self, count: int) -> None:
        super().__init__("head")

        self._count = count

    def resolve_action(self, df: "pd.DataFrame") -> "pd.DataFrame":
        return df.head(self._count)

    def __repr__(self) -> str:
        return f"('{self.type}': ('count': {self._count}))"


class TailAction(PostProcessingAction):
    def __init__(self, count: int) -> None:
        super().__init__("tail")

        self._count = count

    def resolve_action(self, df: "pd.DataFrame") -> "pd.DataFrame":
        return df.tail(self._count)

    def __repr__(self) -> str:
        return f"('{self.type}': ('count': {self._count}))"


class SortFieldAction(PostProcessingAction):
    def __init__(
        self,
        sort_params: List[Dict[str, Dict[str, str]]],
        es_index_fields: Tuple[str, ...],
    ) -> None:
        super().__init__("sort_field")

        self._sort_params = []
        for sort_param in sort_params:
            sort_field, order = sort_param.popitem()
            if sort_field == "_doc":
                sort_field = "_id"
            _, sort_order = order.popitem()
            self._sort_params.append((sort_field, SortOrder.from_string(sort_order)))
        self._es_index_fields = es_index_fields

    def resolve_action(self, df: "pd.DataFrame") -> "pd.DataFrame":
        # Need to detect if a column is also in an index
        # and call sort_index() if that is the case to avoid the
        # ambiguity error raised from sort_values() when there
        # are matches between df.index and df.columns.
        for sort_field, sort_order in self._sort_params[::-1]:
            if sort_field in self._es_index_fields:
                df = df.sort_index(
                    level=self._es_index_fields.index(sort_field),
                    ascending=sort_order == SortOrder.ASC,
                )
            else:
                df = df.sort_values(sort_field, ascending=sort_order == SortOrder.ASC)
        return df

    def __repr__(self) -> str:
        return f"('{self.type}': ('sort_params': '{self._sort_params}'))"
