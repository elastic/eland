from enum import Enum

from pandas.core.indexes.numeric import Int64Index
from pandas.core.indexes.range import RangeIndex

import copy


class Operations:
    """
    A collector of the queries and selectors we apply to queries to return the appropriate results.

    For example,
        - a list of the columns in the DataFrame (a subset of columns in the index)
        - a size limit on the results (e.g. for head(n=5))
        - a query to filter the results (e.g. df.A > 10)

    This is maintained as a 'task graph' (inspired by dask)

    A task graph is a dictionary mapping keys to computations:

    A key is any hashable value that is not a task:
    ```
    {'x': 1,
     'y': 2,
     'z': (add, 'x', 'y'),
     'w': (sum, ['x', 'y', 'z']),
     'v': [(sum, ['w', 'z']), 2]}
    ```
    (see https://docs.dask.org/en/latest/spec.html)
    """

    class SortOrder(Enum):
        ASC = 0
        DESC = 1

        @staticmethod
        def reverse(order):
            if order == Operations.SortOrder.ASC:
                return Operations.SortOrder.DESC

            return Operations.SortOrder.ASC

        @staticmethod
        def to_string(order):
            if order == Operations.SortOrder.ASC:
                return "asc"

            return "desc"

        def from_string(order):
            if order == "asc":
                return Operations.SortOrder.ASC

            return Operations.SortOrder.DESC


    def __init__(self, tasks=None):
        if tasks == None:
            self._tasks = []
        else:
            self._tasks = tasks

    def __constructor__(self, *args, **kwargs):
        return type(self)(*args, **kwargs)

    def copy(self):
        return self.__constructor__(tasks=copy.deepcopy(self._tasks))

    def head(self, index, n):
        # Add a task that is an ascending sort with size=n
        task = ('head', (index.sort_field, n))
        self._tasks.append(task)

    def tail(self, index, n):
        # Add a task that is descending sort with size=n
        task = ('tail', (index.sort_field, n))
        self._tasks.append(task)

    def set_columns(self, columns):
        # Setting columns at different phases of the task list may result in different
        # operations. So instead of setting columns once, set when it happens in call chain
        # TODO - column renaming
        # TODO - validate we are setting columns to a subset of last columns?
        task = ('columns', columns)
        self._tasks.append(task)

    def get_columns(self):
        # Iterate backwards through task list looking for last 'columns' task
        for task in reversed(self._tasks):
            if task[0] == 'columns':
                return task[1]
        return None

    def __repr__(self):
        return repr(self._tasks)

    def to_pandas(self, query_compiler):
        query, post_processing = self._to_es_query()

        size, sort_params = Operations._query_to_params(query)

        # Only return requested columns
        columns = self.get_columns()

        # If size=None use scan not search - then post sort results when in df
        # If size>10000 use scan
        if size is not None and size <= 10000:
            es_results = query_compiler._client.search(
                index=query_compiler._index_pattern,
                size=size,
                sort=sort_params,
                _source=columns)
        else:
            es_results = query_compiler._client.scan(
                index=query_compiler._index_pattern,
                _source=columns)
            # create post sort
            if sort_params is not None:
                post_processing.append(self._sort_params_to_postprocessing(sort_params))

        df = query_compiler._es_results_to_pandas(es_results)

        return self._apply_df_post_processing(df, post_processing)

    def iloc(self, index, columns):
        # index and columns are indexers
        task = ('iloc', (index, columns))
        self._tasks.append(task)

    def squeeze(self, axis):
        task = ('squeeze', axis)
        self._tasks.append(task)

    def to_count(self, query_compiler):
        query, post_processing = self._to_es_query()

        size = query['query_size'] # can be None

        pp_size = self._count_post_processing(post_processing)
        if pp_size is not None:
            if size is not None:
                size = min(size, pp_size)
            else:
                size = pp_size

        # Size is dictated by operations
        if size is not None:
            return size

        exists_query = {"query": {"exists": {"field": query_compiler.index.index_field}}}

        return query_compiler._client.count(index=query_compiler._index_pattern, body=exists_query)

    @staticmethod
    def _sort_params_to_postprocessing(input):
        # Split string
        sort_params = input.split(":")

        query_sort_field = sort_params[0]
        query_sort_order = Operations.SortOrder.from_string(sort_params[1])

        task = ('sort_field', (query_sort_field, query_sort_order))

        return task

    @staticmethod
    def _query_to_params(query):
        sort_params = None
        if query['query_sort_field'] and query['query_sort_order']:
            sort_params = query['query_sort_field'] + ":" + Operations.SortOrder.to_string(query['query_sort_order'])

        size = query['query_size']

        return size, sort_params
    1
    @staticmethod
    def _count_post_processing(post_processing):
        size =  None
        for action in post_processing:
            if action[0] == 'head' or action[0] == 'tail':
                if size is None or action[1][1] < size:
                    size = action[1][1]

        return size

    @staticmethod
    def _apply_df_post_processing(df, post_processing):
        for action in post_processing:
            if action == 'sort_index':
                df = df.sort_index()
            elif action[0] == 'head':
                df = df.head(action[1][1])
            elif action[0] == 'tail':
                df = df.tail(action[1][1])
            elif action[0] == 'sort_field':
                sort_field = action[1][0]
                sort_order = action[1][1]
                if sort_order == Operations.SortOrder.ASC:
                    df = df.sort_values(sort_field, True)
                else:
                    df = df.sort_values(sort_field, False)
            elif action[0] == 'iloc':
                    index_indexer = action[1][0]
                    column_indexer = action[1][1]
                    if index_indexer is None:
                        index_indexer = slice(None)
                    if column_indexer is None:
                        column_indexer = slice(None)
                    df = df.iloc[index_indexer, column_indexer]
            elif action[0] == 'squeeze':
                print(df)
                df = df.squeeze(axis=action[1])
                print(df)

        return df

    def _to_es_query(self):
        # We now try and combine all tasks into an Elasticsearch query
        # Some operations can be simply combined into a single query
        # other operations require pre-queries and then combinations
        # other operations require in-core post-processing of results
        query = {"query_sort_field": None,
                 "query_sort_order": None,
                 "query_size": None,
                 "query_fields": None}

        post_processing = []

        for task in self._tasks:
            if task[0] == 'head':
                query, post_processing = self._resolve_head(task, query, post_processing)
            elif task[0] == 'tail':
                query, post_processing = self._resolve_tail(task, query, post_processing)
            elif task[0] == 'iloc':
                query, post_processing = self._resolve_iloc(task, query, post_processing)
            else: # a lot of operations simply post-process the dataframe - put these straight through
                query, post_processing = self._resolve_post_processing_task(task, query, post_processing)

        return query, post_processing

    def _resolve_head(self, item, query, post_processing):
        # head - sort asc, size n
        # |12345-------------|
        query_sort_field = item[1][0]
        query_sort_order = Operations.SortOrder.ASC
        query_size = item[1][1]

        # If we are already postprocessing the query results, we just get 'head' of these
        # (note, currently we just append another head, we don't optimise by
        # overwriting previous head)
        if len(post_processing) > 0:
            post_processing.append(item)
            return query, post_processing

        if query['query_sort_field'] is None:
            query['query_sort_field'] = query_sort_field
        # if it is already sorted we use existing field

        if query['query_sort_order'] is None:
            query['query_sort_order'] = query_sort_order
        # if it is already sorted we get head of existing order

        if query['query_size'] is None:
            query['query_size'] = query_size
        else:
            # truncate if head is smaller
            if query_size < query['query_size']:
                query['query_size'] = query_size

        return query, post_processing

    def _resolve_tail(self, item, query, post_processing):
        # tail - sort desc, size n, post-process sort asc
        # |-------------12345|
        query_sort_field = item[1][0]
        query_sort_order = Operations.SortOrder.DESC
        query_size = item[1][1]

        # If this is a tail of a tail adjust settings and return
        if query['query_size'] is not None and \
                query['query_sort_order'] == query_sort_order and \
                post_processing == [('sort_index')]:
            if query_size < query['query_size']:
                query['query_size'] = query_size
            return query, post_processing

        # If we are already postprocessing the query results, just get 'tail' of these
        # (note, currently we just append another tail, we don't optimise by
        # overwriting previous tail)
        if len(post_processing) > 0:
            post_processing.append(item)
            return query, post_processing

        # If results are already constrained, just get 'tail' of these
        # (note, currently we just append another tail, we don't optimise by
        # overwriting previous tail)
        if query['query_size'] is not None:
            post_processing.append(item)
            return query, post_processing
        else:
            query['query_size'] = query_size
        if query['query_sort_field'] is None:
            query['query_sort_field'] = query_sort_field
        if query['query_sort_order'] is None:
            query['query_sort_order'] = query_sort_order
        else:
            # reverse sort order
            query['query_sort_order'] = Operations.SortOrder.reverse(query_sort_order)

        post_processing.append(('sort_index'))

        return query, post_processing

    def _resolve_iloc(self, item, query, post_processing):
        # tail - sort desc, size n, post-process sort asc
        # |---4--7-9---------|

        # This is a list of items we return via an integer index
        int_index = item[1][0]
        if int_index is not None:
            last_item = int_index.max()

            # If we have a query_size we do this post processing
            if query['query_size'] is not None:
                post_processing.append(item)
                return query, post_processing

            # size should be > last item
            query['query_size'] = last_item + 1
        post_processing.append(item)

        return query, post_processing

    def _resolve_post_processing_task(self, item, query, post_processing):
        # Just do this in post-processing
        post_processing.append(item)

        return query, post_processing

    def info_es(self, buf):
        buf.write("Operations:\n")
        buf.write("\ttasks: {0}\n".format(self._tasks))

        query, post_processing = self._to_es_query()
        size, sort_params = Operations._query_to_params(query)
        columns = self.get_columns()

        buf.write("\tsize: {0}\n".format(size))
        buf.write("\tsort_params: {0}\n".format(sort_params))
        buf.write("\tcolumns: {0}\n".format(columns))
        buf.write("\tpost_processing: {0}\n".format(post_processing))


