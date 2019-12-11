from abc import ABC, abstractmethod

import numpy as np

from eland import SortOrder
from eland.actions import HeadAction, TailAction, SortIndexAction


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
    def resolve_task(self, query_params, post_processing):
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

    def resolve_task(self, query_params, post_processing):
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

    def resolve_task(self, query_params, post_processing):
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

    def resolve_task(self, query_params, post_processing):
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

    def resolve_task(self, query_params, post_processing):
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

    def resolve_task(self, query_params, post_processing):
        query_params['query'].update_boolean_filter(self._boolean_filter)

        return query_params, post_processing


class ArithmeticOpFieldsTask(Task):
    def __init__(self, field_name, op_name, left_field, right_field, op_type):
        super().__init__("arithmetic_op_fields")

        self._field_name = field_name
        self._op_name = op_name
        self._left_field = left_field
        self._right_field = right_field
        self._op_type = op_type

    def __repr__(self):
        return "('{}': (" \
               "'field_name': {}, " \
               "'op_name': {}, " \
               "'left_field': {}, " \
               "'right_field': {}, " \
               "'op_type': {}" \
               "))" \
            .format(self._task_type, self._field_name, self._op_name, self._left_field, self._right_field,
                    self._op_type)

    def resolve_task(self, query_params, post_processing):
        # https://www.elastic.co/guide/en/elasticsearch/painless/current/painless-api-reference-shared-java-lang.html#painless-api-reference-shared-Math
        if not self._op_type:
            if isinstance(self._left_field, str) and isinstance(self._right_field, str):
                """
                (if op_name = '__truediv__')

                "script_fields": {
                    "field_name": {
                    "script": {
                        "source": "doc[left_field].value / doc[right_field].value"
                    }
                    }
                }
                """
                if self._op_name == '__add__':
                    source = "doc['{0}'].value + doc['{1}'].value".format(self._left_field, self._right_field)
                elif self._op_name == '__truediv__':
                    source = "doc['{0}'].value / doc['{1}'].value".format(self._left_field, self._right_field)
                elif self._op_name == '__floordiv__':
                    source = "Math.floor(doc['{0}'].value / doc['{1}'].value)".format(self._left_field,
                                                                                      self._right_field)
                elif self._op_name == '__pow__':
                    source = "Math.pow(doc['{0}'].value, doc['{1}'].value)".format(self._left_field, self._right_field)
                elif self._op_name == '__mod__':
                    source = "doc['{0}'].value % doc['{1}'].value".format(self._left_field, self._right_field)
                elif self._op_name == '__mul__':
                    source = "doc['{0}'].value * doc['{1}'].value".format(self._left_field, self._right_field)
                elif self._op_name == '__sub__':
                    source = "doc['{0}'].value - doc['{1}'].value".format(self._left_field, self._right_field)
                else:
                    raise NotImplementedError("Not implemented operation '{0}'".format(self._op_name))

                if query_params['query_script_fields'] is None:
                    query_params['query_script_fields'] = dict()
                query_params['query_script_fields'][self._field_name] = {
                    'script': {
                        'source': source
                    }
                }
            elif isinstance(self._left_field, str) and np.issubdtype(np.dtype(type(self._right_field)), np.number):
                """
                (if self._op_name = '__truediv__')

                "script_fields": {
                    "field_name": {
                    "script": {
                        "source": "doc[self._left_field].value / self._right_field"
                    }
                    }
                }
                """
                if self._op_name == '__add__':
                    source = "doc['{0}'].value + {1}".format(self._left_field, self._right_field)
                elif self._op_name == '__truediv__':
                    source = "doc['{0}'].value / {1}".format(self._left_field, self._right_field)
                elif self._op_name == '__floordiv__':
                    source = "Math.floor(doc['{0}'].value / {1})".format(self._left_field, self._right_field)
                elif self._op_name == '__pow__':
                    source = "Math.pow(doc['{0}'].value, {1})".format(self._left_field, self._right_field)
                elif self._op_name == '__mod__':
                    source = "doc['{0}'].value % {1}".format(self._left_field, self._right_field)
                elif self._op_name == '__mul__':
                    source = "doc['{0}'].value * {1}".format(self._left_field, self._right_field)
                elif self._op_name == '__sub__':
                    source = "doc['{0}'].value - {1}".format(self._left_field, self._right_field)
                else:
                    raise NotImplementedError("Not implemented operation '{0}'".format(self._op_name))
            elif np.issubdtype(np.dtype(type(self._left_field)), np.number) and isinstance(self._right_field, str):
                """
                (if self._op_name = '__truediv__')

                "script_fields": {
                    "field_name": {
                    "script": {
                        "source": "self._left_field / doc['self._right_field'].value"
                    }
                    }
                }
                """
                if self._op_name == '__add__':
                    source = "{0} + doc['{1}'].value".format(self._left_field, self._right_field)
                elif self._op_name == '__truediv__':
                    source = "{0} / doc['{1}'].value".format(self._left_field, self._right_field)
                elif self._op_name == '__floordiv__':
                    source = "Math.floor({0} / doc['{1}'].value)".format(self._left_field, self._right_field)
                elif self._op_name == '__pow__':
                    source = "Math.pow({0}, doc['{1}'].value)".format(self._left_field, self._right_field)
                elif self._op_name == '__mod__':
                    source = "{0} % doc['{1}'].value".format(self._left_field, self._right_field)
                elif self._op_name == '__mul__':
                    source = "{0} * doc['{1}'].value".format(self._left_field, self._right_field)
                elif self._op_name == '__sub__':
                    source = "{0} - doc['{1}'].value".format(self._left_field, self._right_field)
                else:
                    raise NotImplementedError("Not implemented operation '{0}'".format(self._op_name))

            else:
                raise TypeError("Types for operation inconsistent {} {} {}", type(self._left_field),
                                type(self._right_field), self._op_name)

        elif self._op_type[0] == "string":
            # we need to check the type of string addition
            if self._op_type[1] == "s":
                """
                (if self._op_name = '__add__')

                "script_fields": {
                    "field_name": {
                    "script": {
                        "source": "doc[self._left_field].value + doc[self._right_field].value"
                    }
                    }
                }
                """
                if self._op_name == '__add__':
                    source = "doc['{0}'].value + doc['{1}'].value".format(self._left_field, self._right_field)
                else:
                    raise NotImplementedError("Not implemented operation '{0}'".format(self._op_name))

            elif self._op_type[1] == "r":
                if isinstance(self._left_field, str) and isinstance(self._right_field, str):
                    """
                    (if self._op_name = '__add__')

                    "script_fields": {
                        "field_name": {
                        "script": {
                            "source": "doc[self._left_field].value + self._right_field"
                        }
                        }
                    }
                    """
                    if self._op_name == '__add__':
                        source = "doc['{0}'].value + '{1}'".format(self._left_field, self._right_field)
                    else:
                        raise NotImplementedError("Not implemented operation '{0}'".format(self._op_name))

            elif self._op_type[1] == 'l':
                if isinstance(self._left_field, str) and isinstance(self._right_field, str):
                    """
                    (if self._op_name = '__add__')

                    "script_fields": {
                        "field_name": {
                        "script": {
                            "source": "self._left_field + doc[self._right_field].value"
                        }
                        }
                    }
                    """
                    if self._op_name == '__add__':
                        source = "'{0}' + doc['{1}'].value".format(self._left_field, self._right_field)
                    else:
                        raise NotImplementedError("Not implemented operation '{0}'".format(self._op_name))

        if query_params['query_script_fields'] is None:
            query_params['query_script_fields'] = dict()
        query_params['query_script_fields'][self._field_name] = {
            'script': {
                'source': source
            }
        }

        return query_params, post_processing
