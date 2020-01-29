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

# Default number of rows displayed (different to pandas where ALL could be displayed)
from enum import Enum

DEFAULT_NUM_ROWS_DISPLAYED = 60

DEFAULT_CHUNK_SIZE = 10000
DEFAULT_CSV_BATCH_OUTPUT_SIZE = 10000
DEFAULT_PROGRESS_REPORTING_NUM_ROWS = 10000
DEFAULT_ES_MAX_RESULT_WINDOW = 10000  # index.max_result_window


def docstring_parameter(*sub):
    def dec(obj):
        obj.__doc__ = obj.__doc__.format(*sub)
        return obj

    return dec


class SortOrder(Enum):
    ASC = 0
    DESC = 1

    @staticmethod
    def reverse(order):
        if order == SortOrder.ASC:
            return SortOrder.DESC

        return SortOrder.ASC

    @staticmethod
    def to_string(order):
        if order == SortOrder.ASC:
            return "asc"

        return "desc"

    @staticmethod
    def from_string(order):
        if order == "asc":
            return SortOrder.ASC

        return SortOrder.DESC
