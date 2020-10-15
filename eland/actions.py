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
from typing import TYPE_CHECKING, List, Optional, Union

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
    def __init__(self, items: Optional[Union[List[int], List[str]]] = None) -> None:
        super().__init__("sort_index")
        self._items = items

    def resolve_action(self, df: "pd.DataFrame") -> "pd.DataFrame":
        if self._items is not None:
            return df.reindex(self._items)
        else:
            return df.sort_index()

    def __repr__(self) -> str:
        return f"('{self.type}')"


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
    def __init__(self, sort_params_string: str) -> None:
        super().__init__("sort_field")

        if sort_params_string is None:
            raise ValueError("Expected valid string")

        # Split string
        sort_field, _, sort_order = sort_params_string.partition(":")
        if not sort_field or sort_order not in ("asc", "desc"):
            raise ValueError(
                f"Expected ES sort params string (e.g. _doc:desc). Got '{sort_params_string}'"
            )

        self._sort_field = sort_field
        self._sort_order = SortOrder.from_string(sort_order)

    def resolve_action(self, df: "pd.DataFrame") -> "pd.DataFrame":
        return df.sort_values(self._sort_field, self._sort_order == SortOrder.ASC)

    def __repr__(self) -> str:
        return f"('{self.type}': ('sort_field': '{self._sort_field}', 'sort_order': {self._sort_order}))"
