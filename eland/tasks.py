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

from eland import SortOrder
from eland.actions import HeadAction, TailAction, SortIndexAction
from eland.arithmetics import ArithmeticSeries


# -------------------------------------------------------------------------------------------------------------------- #
# Tasks                                                                                                                #
# -------------------------------------------------------------------------------------------------------------------- #
class Task(ABC):
    """
    Abstract class for tasks

    Parameters
    ----------
        task_type: str
            The task type (e.g. head, tail etc.)
    """

    def __init__(self, task_type):
        self._task_type = task_type

    @property
    def type(self):
        return self._task_type

    @abstractmethod
    def resolve_task(self, query_params, post_processing, query_compiler):
        pass

    @abstractmethod
    def __repr__(self):
        pass


class SizeTask(Task):
    def __init__(self, task_type):
        super().__init__(task_type)

    @abstractmethod
    def size(self):
        # must override
        pass


class HeadTask(SizeTask):
    def __init__(self, sort_field, count):
        super().__init__("head")

        # Add a task that is an ascending sort with size=count
        self._sort_field = sort_field
        self._count = count

    def __repr__(self):
        return "('{}': ('sort_field': '{}', 'count': {}))".format(self._task_type, self._sort_field, self._count)

    def resolve_task(self, query_params, post_processing, query_compiler):
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

        if query_params['query_sort_field'] is None:
            query_params['query_sort_field'] = query_sort_field
        # if it is already sorted we use existing field

        if query_params['query_sort_order'] is None:
            query_params['query_sort_order'] = query_sort_order
        # if it is already sorted we get head of existing order

        if query_params['query_size'] is None:
            query_params['query_size'] = query_size
        else:
            # truncate if head is smaller
            if query_size < query_params['query_size']:
                query_params['query_size'] = query_size

        return query_params, post_processing

    def size(self):
        return self._count


class TailTask(SizeTask):
    def __init__(self, sort_field, count):
        super().__init__("tail")

        # Add a task that is descending sort with size=count
        self._sort_field = sort_field
        self._count = count

    def __repr__(self):
        return "('{}': ('sort_field': '{}', 'count': {}))".format(self._task_type, self._sort_field, self._count)

    def resolve_task(self, query_params, post_processing, query_compiler):
        # tail - sort desc, size n, post-process sort asc
        # |-------------12345|
        query_sort_field = self._sort_field
        query_sort_order = SortOrder.DESC
        query_size = self._count

        # If this is a tail of a tail adjust settings and return
        if query_params['query_size'] is not None and \
                query_params['query_sort_order'] == query_sort_order and \
                post_processing == ['sort_index']:
            if query_size < query_params['query_size']:
                query_params['query_size'] = query_size
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
        if query_params['query_size'] is not None:
            post_processing.append(TailAction(self._count))
            return query_params, post_processing
        else:
            query_params['query_size'] = query_size
        if query_params['query_sort_field'] is None:
            query_params['query_sort_field'] = query_sort_field
        if query_params['query_sort_order'] is None:
            query_params['query_sort_order'] = query_sort_order
        else:
            # reverse sort order
            query_params['query_sort_order'] = SortOrder.reverse(query_sort_order)

        post_processing.append(SortIndexAction())

        return query_params, post_processing

    def size(self):
        return self._count


class QueryIdsTask(Task):
    def __init__(self, must, ids):
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

    def resolve_task(self, query_params, post_processing, query_compiler):
        query_params['query'].ids(self._ids, must=self._must)

        return query_params, post_processing

    def __repr__(self):
        return "('{}': ('must': {}, 'ids': {}))".format(self._task_type, self._must, self._ids)


class QueryTermsTask(Task):
    def __init__(self, must, field, terms):
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

    def __repr__(self):
        return "('{}': ('must': {}, 'field': '{}', 'terms': {}))".format(self._task_type, self._must, self._field,
                                                                         self._terms)

    def resolve_task(self, query_params, post_processing, query_compiler):
        query_params['query'].terms(self._field, self._terms, must=self._must)

        return query_params, post_processing


class BooleanFilterTask(Task):
    def __init__(self, boolean_filter):
        """
        Parameters
        ----------
        boolean_filter: BooleanFilter or str
            The filter to apply
        """
        super().__init__("boolean_filter")

        self._boolean_filter = boolean_filter

    def __repr__(self):
        return "('{}': ('boolean_filter': {}))".format(self._task_type, repr(self._boolean_filter))

    def resolve_task(self, query_params, post_processing, query_compiler):
        query_params['query'].update_boolean_filter(self._boolean_filter)

        return query_params, post_processing


class ArithmeticOpFieldsTask(Task):
    def __init__(self, display_name, arithmetic_series):
        super().__init__("arithmetic_op_fields")

        self._display_name = display_name

        if not isinstance(arithmetic_series, ArithmeticSeries):
            raise TypeError("Expecting ArithmeticSeries got {}".format(type(arithmetic_series)))
        self._arithmetic_series = arithmetic_series

    def __repr__(self):
        return "('{}': (" \
               "'display_name': {}, " \
               "'arithmetic_object': {}" \
               "))" \
            .format(self._task_type, self._display_name, self._arithmetic_series)

    def update(self, display_name, arithmetic_series):
        self._display_name = display_name
        self._arithmetic_series = arithmetic_series

    def resolve_task(self, query_params, post_processing, query_compiler):
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
        if query_params['query_script_fields'] is None:
            query_params['query_script_fields'] = dict()

        if self._display_name in query_params['query_script_fields']:
            raise NotImplementedError(
                "TODO code path - combine multiple ops '{}'\n{}\n{}\n{}".format(self,
                                                                                query_params['query_script_fields'],
                                                                                self._display_name,
                                                                                self._arithmetic_series.resolve()))

        query_params['query_script_fields'][self._display_name] = {
            'script': {
                'source': self._arithmetic_series.resolve()
            }
        }

        return query_params, post_processing
