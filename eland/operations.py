from enum import Enum


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
                return ":asc"

            return ":desc"

    def __init__(self, tasks=None):
        if tasks == None:
            self._tasks = []
        else:
            self._tasks = tasks

    def __constructor__(self, *args, **kwargs):
        return type(self)(*args, **kwargs)

    def copy(self):
        return self.__constructor__(tasks=self._tasks.copy())

    def head(self, index, n):
        # Add a task that is an ascending sort with size=n
        task = ('head', (index.sort_field, n))
        self._tasks.append(task)

    def tail(self, index, n):

        # Add a task that is descending sort with size=n
        task = ('tail', (index.sort_field, n))
        self._tasks.append(task)

    def set_columns(self, columns):
        self._tasks['columns'] = columns

    def __repr__(self):
        return repr(self._tasks)

    def to_pandas(self, query_compiler):
        query, post_processing = self._to_es_query()

        size, sort_params = Operations._query_to_params(query)

        es_results = query_compiler._client.search(
            index=query_compiler._index_pattern,
            size=size,
            sort=sort_params)

        df = query_compiler._es_results_to_pandas(es_results)

        return self._apply_df_post_processing(df, post_processing)

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
    def _query_to_params(query):
        sort_params = None
        if query['query_sort_field'] and query['query_sort_order']:
            sort_params = query['query_sort_field'] + Operations.SortOrder.to_string(query['query_sort_order'])

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
            print(action)
            if action == 'sort_index':
                df = df.sort_index()
            elif action[0] == 'head':
                df = df.head(action[1][1])
            elif action[0] == 'tail':
                df = df.tail(action[1][1])

        return df

    def _to_es_query(self):
        # We now try and combine all tasks into an Elasticsearch query
        # Some operations can be simply combined into a single query
        # other operations require pre-queries and then combinations
        # other operations require in-core post-processing of results
        query = {"query_sort_field": None,
                 "query_sort_order": None,
                 "query_size": None}

        post_processing = []

        for task in self._tasks:
            if task[0] == 'head':
                query, post_processing = self._resolve_head(task, query, post_processing)
            elif task[0] == 'tail':
                query, post_processing = self._resolve_tail(task, query, post_processing)

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
