#  Copyright 2019 Elasticsearch BV
#
#      Licensed under the Apache License, Version 2.0 (the "License");
#      you may not use this file except in compliance with the License.
#      You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#      Unless required by applicable law or agreed to in writing, software
#      distributed under the License is distributed on an "AS IS" BASIS,
#      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#      See the License for the specific language governing permissions and
#      limitations under the License.

from abc import ABC, abstractmethod

# -------------------------------------------------------------------------------------------------------------------- #
# PostProcessingActions                                                                                                #
# -------------------------------------------------------------------------------------------------------------------- #
from eland import SortOrder


class PostProcessingAction(ABC):
    def __init__(self, action_type):
        """
        Abstract class for postprocessing actions

        Parameters
        ----------
            action_type: str
                The action type (e.g. sort_index, head etc.)
        """
        self._action_type = action_type

    @property
    def type(self):
        return self._action_type

    @abstractmethod
    def resolve_action(self, df):
        pass

    @abstractmethod
    def __repr__(self):
        pass


class SortIndexAction(PostProcessingAction):
    def __init__(self):
        super().__init__("sort_index")

    def resolve_action(self, df):
        return df.sort_index()

    def __repr__(self):
        return "('{}')".format(self.type)


class HeadAction(PostProcessingAction):
    def __init__(self, count):
        super().__init__("head")

        self._count = count

    def resolve_action(self, df):
        return df.head(self._count)

    def __repr__(self):
        return "('{}': ('count': {}))".format(self.type, self._count)


class TailAction(PostProcessingAction):
    def __init__(self, count):
        super().__init__("tail")

        self._count = count

    def resolve_action(self, df):
        return df.tail(self._count)

    def __repr__(self):
        return "('{}': ('count': {}))".format(self.type, self._count)


class SortFieldAction(PostProcessingAction):
    def __init__(self, sort_params_string):
        super().__init__("sort_field")

        if sort_params_string is None:
            raise ValueError("Expected valid string")

        # Split string
        sort_params = sort_params_string.split(":")
        if len(sort_params) != 2:
            raise ValueError("Expected ES sort params string (e.g. _doc:desc). Got '{}'".format(sort_params_string))

        self._sort_field = sort_params[0]
        self._sort_order = SortOrder.from_string(sort_params[1])

    def resolve_action(self, df):
        if self._sort_order == SortOrder.ASC:
            return df.sort_values(self._sort_field, True)
        return df.sort_values(self._sort_field, False)

    def __repr__(self):
        return "('{}': ('sort_field': '{}', 'sort_order': {}))".format(self.type, self._sort_field, self._sort_order)
