import copy
from enum import Enum

import numpy as np
import pandas as pd

from eland import Index
from eland import Query


class Operations:
    """
    A collector of the queries and selectors we apply to queries to return the appropriate results.

    For example,
        - a list of the field_names in the DataFrame (a subset of field_names in the index)
        - a size limit on the results (e.g. for head(n=5))
        - a query to filter the results (e.g. df.A > 10)

    This is maintained as a 'task graph' (inspired by dask)
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

        @staticmethod
        def from_string(order):
            if order == "asc":
                return Operations.SortOrder.ASC

            return Operations.SortOrder.DESC

    def __init__(self, tasks=None):
        if tasks is None:
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

    def arithmetic_op_fields(self, field_name, op_name, left_field, right_field, op_type=None):
        if op_type:
            task = ('arithmetic_op_fields', (field_name, (op_name, (left_field, right_field))), op_type)
        else:
            task = ('arithmetic_op_fields', (field_name, (op_name, (left_field, right_field))))
        # Set this as a column we want to retrieve
        self.set_field_names([field_name])

        self._tasks.append(task)

    def set_field_names(self, field_names):
        # Setting field_names at different phases of the task list may result in different
        # operations. So instead of setting field_names once, set when it happens in call chain
        if not isinstance(field_names, list):
            field_names = list(field_names)

        # TODO - field_name renaming
        # TODO - validate we are setting field_names to a subset of last field_names?
        task = ('field_names', field_names)
        self._tasks.append(task)
        # Iterate backwards through task list looking for last 'field_names' task
        for task in reversed(self._tasks):
            if task[0] == 'field_names':
                return task[1]
        return None

    def get_field_names(self):
        # Iterate backwards through task list looking for last 'field_names' task
        for task in reversed(self._tasks):
            if task[0] == 'field_names':
                return task[1]
        return None

    def __repr__(self):
        return repr(self._tasks)

    def count(self, query_compiler):
        query_params, post_processing = self._resolve_tasks()

        # Elasticsearch _count is very efficient and so used to return results here. This means that
        # data frames that have restricted size or sort params will not return valid results
        # (_count doesn't support size).
        # Longer term we may fall back to pandas, but this may result in loading all index into memory.
        if self._size(query_params, post_processing) is not None:
            raise NotImplementedError("Requesting count with additional query and processing parameters "
                                      "not supported {0} {1}"
                                      .format(query_params, post_processing))

        # Only return requested field_names
        fields = query_compiler.field_names

        counts = {}
        for field in fields:
            body = Query(query_params['query'])
            body.exists(field, must=True)

            field_exists_count = query_compiler._client.count(index=query_compiler._index_pattern,
                                                              body=body.to_count_body())
            counts[field] = field_exists_count

        return pd.Series(data=counts, index=fields)

    def mean(self, query_compiler):
        return self._metric_aggs(query_compiler, 'avg')

    def sum(self, query_compiler):
        return self._metric_aggs(query_compiler, 'sum')

    def max(self, query_compiler):
        return self._metric_aggs(query_compiler, 'max')

    def min(self, query_compiler):
        return self._metric_aggs(query_compiler, 'min')

    def nunique(self, query_compiler):
        return self._metric_aggs(query_compiler, 'cardinality', field_types='aggregatable')

    def value_counts(self, query_compiler, es_size):
        return self._terms_aggs(query_compiler, 'terms', es_size)

    def hist(self, query_compiler, bins):
        return self._hist_aggs(query_compiler, bins)

    def _metric_aggs(self, query_compiler, func, field_types=None):
        """
        Parameters
        ----------
        field_types: str, default None
            if `aggregatable` use only field_names whose fields in elasticseach are aggregatable.
            If `None`, use only numeric fields.

        Returns
        -------
        pandas.Series
            Series containing results of `func` applied to the field_name(s)
        """
        query_params, post_processing = self._resolve_tasks()

        size = self._size(query_params, post_processing)
        if size is not None:
            raise NotImplementedError("Can not count field matches if size is set {}".format(size))

        field_names = self.get_field_names()

        body = Query(query_params['query'])

        # some metrics aggs (including cardinality) work on all aggregatable fields
        # therefore we include an optional all parameter on operations
        # that call _metric_aggs
        if field_types == 'aggregatable':
            source_fields = query_compiler._mappings.aggregatable_field_names(field_names)
        else:
            source_fields = query_compiler._mappings.numeric_source_fields(field_names)

        for field in source_fields:
            body.metric_aggs(field, func, field)

        response = query_compiler._client.search(
            index=query_compiler._index_pattern,
            size=0,
            body=body.to_search_body())

        # Results are of the form
        # "aggregations" : {
        #   "AvgTicketPrice" : {
        #     "value" : 628.2536888148849
        #   }
        # }
        results = {}

        if field_types == 'aggregatable':
            for key, value in source_fields.items():
                results[value] = response['aggregations'][key]['value']
        else:
            for field in source_fields:
                results[field] = response['aggregations'][field]['value']

        # Return single value if this is a series
        # if len(numeric_source_fields) == 1:
        #    return np.float64(results[numeric_source_fields[0]])
        s = pd.Series(data=results, index=results.keys())

        return s

    def _terms_aggs(self, query_compiler, func, es_size=None):
        """
        Parameters
        ----------
        es_size: int, default None
            Parameter used by Series.value_counts()

        Returns
        -------
        pandas.Series
            Series containing results of `func` applied to the field_name(s)
        """
        query_params, post_processing = self._resolve_tasks()

        size = self._size(query_params, post_processing)
        if size is not None:
            raise NotImplementedError("Can not count field matches if size is set {}".format(size))

        field_names = self.get_field_names()

        # Get just aggregatable field_names
        aggregatable_field_names = query_compiler._mappings.aggregatable_field_names(field_names)

        body = Query(query_params['query'])

        for field in aggregatable_field_names.keys():
            body.terms_aggs(field, func, field, es_size=es_size)

        response = query_compiler._client.search(
            index=query_compiler._index_pattern,
            size=0,
            body=body.to_search_body())

        results = {}

        for key, value in aggregatable_field_names.items():
            for bucket in response['aggregations'][field_names[0]]['buckets']:
                results[bucket['key']] = bucket['doc_count']

        try:
            name = field_names[0]
        except IndexError:
            name = None

        s = pd.Series(data=results, index=results.keys(), name=name)

        return s

    def _hist_aggs(self, query_compiler, num_bins):
        # Get histogram bins and weights for numeric field_names
        query_params, post_processing = self._resolve_tasks()

        size = self._size(query_params, post_processing)
        if size is not None:
            raise NotImplementedError("Can not count field matches if size is set {}".format(size))

        field_names = self.get_field_names()

        numeric_source_fields = query_compiler._mappings.numeric_source_fields(field_names)

        body = Query(query_params['query'])

        min_aggs = self._metric_aggs(query_compiler, 'min')
        max_aggs = self._metric_aggs(query_compiler, 'max')

        for field in numeric_source_fields:
            body.hist_aggs(field, field, min_aggs, max_aggs, num_bins)

        response = query_compiler._client.search(
            index=query_compiler._index_pattern,
            size=0,
            body=body.to_search_body())

        # results are like
        # "aggregations" : {
        #     "DistanceKilometers" : {
        #       "buckets" : [
        #         {
        #           "key" : 0.0,
        #           "doc_count" : 2956
        #         },
        #         {
        #           "key" : 1988.1482421875,
        #           "doc_count" : 768
        #         },
        #         ...

        bins = {}
        weights = {}

        # There is one more bin that weights
        # len(bins) = len(weights) + 1

        # bins = [  0.  36.  72. 108. 144. 180. 216. 252. 288. 324. 360.]
        # len(bins) == 11
        # weights = [10066.,   263.,   386.,   264.,   273.,   390.,   324.,   438.,   261.,   394.]
        # len(weights) == 10

        # ES returns
        # weights = [10066.,   263.,   386.,   264.,   273.,   390.,   324.,   438.,   261.,   252.,    142.]
        # So sum last 2 buckets
        for field in numeric_source_fields:
            buckets = response['aggregations'][field]['buckets']

            bins[field] = []
            weights[field] = []

            for bucket in buckets:
                bins[field].append(bucket['key'])

                if bucket == buckets[-1]:
                    weights[field][-1] += bucket['doc_count']
                else:
                    weights[field].append(bucket['doc_count'])

        df_bins = pd.DataFrame(data=bins)
        df_weights = pd.DataFrame(data=weights)

        return df_bins, df_weights

    @staticmethod
    def _map_pd_aggs_to_es_aggs(pd_aggs):
        """
        Args:
            pd_aggs - list of pandas aggs (e.g. ['mad', 'min', 'std'] etc.)

        Returns:
            ed_aggs - list of corresponding es_aggs (e.g. ['median_absolute_deviation', 'min', 'std'] etc.)

        Pandas supports a lot of options here, and these options generally work on text and numerics in pandas.
        Elasticsearch has metric aggs and terms aggs so will have different behaviour.

        Pandas aggs that return field_names (as opposed to transformed rows):

        all
        any
        count
        mad
        max
        mean
        median
        min
        mode
        quantile
        rank
        sem
        skew
        sum
        std
        var
        nunique
        """
        ed_aggs = []
        for pd_agg in pd_aggs:
            if pd_agg == 'count':
                ed_aggs.append('count')
            elif pd_agg == 'mad':
                ed_aggs.append('median_absolute_deviation')
            elif pd_agg == 'max':
                ed_aggs.append('max')
            elif pd_agg == 'mean':
                ed_aggs.append('avg')
            elif pd_agg == 'median':
                ed_aggs.append(('percentiles', '50.0'))
            elif pd_agg == 'min':
                ed_aggs.append('min')
            elif pd_agg == 'mode':
                # We could do this via top term
                raise NotImplementedError(pd_agg, " not currently implemented")
            elif pd_agg == 'quantile':
                # TODO
                raise NotImplementedError(pd_agg, " not currently implemented")
            elif pd_agg == 'rank':
                # TODO
                raise NotImplementedError(pd_agg, " not currently implemented")
            elif pd_agg == 'sem':
                # TODO
                raise NotImplementedError(pd_agg, " not currently implemented")
            elif pd_agg == 'sum':
                ed_aggs.append('sum')
            elif pd_agg == 'std':
                ed_aggs.append(('extended_stats', 'std_deviation'))
            elif pd_agg == 'var':
                ed_aggs.append(('extended_stats', 'variance'))
            else:
                raise NotImplementedError(pd_agg, " not currently implemented")

        # TODO - we can optimise extended_stats here as if we have 'count' and 'std' extended_stats would
        #   return both in one call

        return ed_aggs

    def aggs(self, query_compiler, pd_aggs):
        query_params, post_processing = self._resolve_tasks()

        size = self._size(query_params, post_processing)
        if size is not None:
            raise NotImplementedError("Can not count field matches if size is set {}".format(size))

        field_names = self.get_field_names()

        body = Query(query_params['query'])

        # convert pandas aggs to ES equivalent
        es_aggs = self._map_pd_aggs_to_es_aggs(pd_aggs)

        for field in field_names:
            for es_agg in es_aggs:
                # If we have multiple 'extended_stats' etc. here we simply NOOP on 2nd call
                if isinstance(es_agg, tuple):
                    body.metric_aggs(es_agg[0] + '_' + field, es_agg[0], field)
                else:
                    body.metric_aggs(es_agg + '_' + field, es_agg, field)

        response = query_compiler._client.search(
            index=query_compiler._index_pattern,
            size=0,
            body=body.to_search_body())

        """
        Results are like (for 'sum', 'min')

             AvgTicketPrice  DistanceKilometers  DistanceMiles  FlightDelayMin
        sum    8.204365e+06        9.261629e+07   5.754909e+07          618150
        min    1.000205e+02        0.000000e+00   0.000000e+00               0
        """
        results = {}

        for field in field_names:
            values = list()
            for es_agg in es_aggs:
                if isinstance(es_agg, tuple):
                    values.append(response['aggregations'][es_agg[0] + '_' + field][es_agg[1]])
                else:
                    values.append(response['aggregations'][es_agg + '_' + field]['value'])

            results[field] = values

        df = pd.DataFrame(data=results, index=pd_aggs)

        return df

    def describe(self, query_compiler):
        query_params, post_processing = self._resolve_tasks()

        size = self._size(query_params, post_processing)
        if size is not None:
            raise NotImplementedError("Can not count field matches if size is set {}".format(size))

        field_names = self.get_field_names()

        numeric_source_fields = query_compiler._mappings.numeric_source_fields(field_names, include_bool=False)

        # for each field we compute:
        # count, mean, std, min, 25%, 50%, 75%, max
        body = Query(query_params['query'])

        for field in numeric_source_fields:
            body.metric_aggs('extended_stats_' + field, 'extended_stats', field)
            body.metric_aggs('percentiles_' + field, 'percentiles', field)

        response = query_compiler._client.search(
            index=query_compiler._index_pattern,
            size=0,
            body=body.to_search_body())

        results = {}

        for field in numeric_source_fields:
            values = list()
            values.append(response['aggregations']['extended_stats_' + field]['count'])
            values.append(response['aggregations']['extended_stats_' + field]['avg'])
            values.append(response['aggregations']['extended_stats_' + field]['std_deviation'])
            values.append(response['aggregations']['extended_stats_' + field]['min'])
            values.append(response['aggregations']['percentiles_' + field]['values']['25.0'])
            values.append(response['aggregations']['percentiles_' + field]['values']['50.0'])
            values.append(response['aggregations']['percentiles_' + field]['values']['75.0'])
            values.append(response['aggregations']['extended_stats_' + field]['max'])

            # if not None
            if values.count(None) < len(values):
                results[field] = values

        df = pd.DataFrame(data=results, index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])

        return df

    def to_pandas(self, query_compiler):
        class PandasDataFrameCollector:
            def __init__(self):
                self.df = None

            def collect(self, df):
                self.df = df

            @staticmethod
            def batch_size():
                return None

        collector = PandasDataFrameCollector()

        self._es_results(query_compiler, collector)

        return collector.df

    def to_csv(self, query_compiler, **kwargs):
        class PandasToCSVCollector:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.ret = None
                self.first_time = True

            def collect(self, df):
                # If this is the first time we collect results, then write header, otherwise don't write header
                # and append results
                if self.first_time:
                    self.first_time = False
                    df.to_csv(**self.kwargs)
                else:
                    # Don't write header, and change mode to append
                    self.kwargs['header'] = False
                    self.kwargs['mode'] = 'a'
                    df.to_csv(**self.kwargs)

            @staticmethod
            def batch_size():
                # By default read 10000 docs to csv
                batch_size = 10000
                return batch_size

        collector = PandasToCSVCollector(**kwargs)

        self._es_results(query_compiler, collector)

        return collector.ret

    def _es_results(self, query_compiler, collector):
        query_params, post_processing = self._resolve_tasks()

        size, sort_params = Operations._query_params_to_size_and_sort(query_params)

        script_fields = query_params['query_script_fields']
        query = Query(query_params['query'])

        body = query.to_search_body()
        if script_fields is not None:
            body['script_fields'] = script_fields

        # Only return requested field_names
        field_names = self.get_field_names()

        es_results = None

        # If size=None use scan not search - then post sort results when in df
        # If size>10000 use scan
        is_scan = False
        if size is not None and size <= 10000:
            if size > 0:
                try:
                    es_results = query_compiler._client.search(
                        index=query_compiler._index_pattern,
                        size=size,
                        sort=sort_params,
                        body=body,
                        _source=field_names)
                except Exception:
                    # Catch all ES errors and print debug (currently to stdout)
                    error = {
                        'index': query_compiler._index_pattern,
                        'size': size,
                        'sort': sort_params,
                        'body': body,
                        '_source': field_names
                    }
                    print("Elasticsearch error:", error)
                    raise
        else:
            is_scan = True
            es_results = query_compiler._client.scan(
                index=query_compiler._index_pattern,
                query=body,
                _source=field_names)
            # create post sort
            if sort_params is not None:
                post_processing.append(self._sort_params_to_postprocessing(sort_params))

        if is_scan:
            while True:
                partial_result, df = query_compiler._es_results_to_pandas(es_results, collector.batch_size())
                df = self._apply_df_post_processing(df, post_processing)
                collector.collect(df)
                if not partial_result:
                    break
        else:
            partial_result, df = query_compiler._es_results_to_pandas(es_results)
            df = self._apply_df_post_processing(df, post_processing)
            collector.collect(df)

    def iloc(self, index, field_names):
        # index and field_names are indexers
        task = ('iloc', (index, field_names))
        self._tasks.append(task)

    def index_count(self, query_compiler, field):
        # field is the index field so count values
        query_params, post_processing = self._resolve_tasks()

        size = self._size(query_params, post_processing)

        # Size is dictated by operations
        if size is not None:
            # TODO - this is not necessarily valid as the field may not exist in ALL these docs
            return size

        body = Query(query_params['query'])
        body.exists(field, must=True)

        return query_compiler._client.count(index=query_compiler._index_pattern, body=body.to_count_body())

    def _validate_index_operation(self, items):
        if not isinstance(items, list):
            raise TypeError("list item required - not {}".format(type(items)))

        # field is the index field so count values
        query_params, post_processing = self._resolve_tasks()

        size = self._size(query_params, post_processing)

        # Size is dictated by operations
        if size is not None:
            raise NotImplementedError("Can not count field matches if size is set {}".format(size))

        return query_params, post_processing

    def index_matches_count(self, query_compiler, field, items):
        query_params, post_processing = self._validate_index_operation(items)

        body = Query(query_params['query'])

        if field == Index.ID_INDEX_FIELD:
            body.ids(items, must=True)
        else:
            body.terms(field, items, must=True)

        return query_compiler._client.count(index=query_compiler._index_pattern, body=body.to_count_body())

    def drop_index_values(self, query_compiler, field, items):
        self._validate_index_operation(items)

        # Putting boolean queries together
        # i = 10
        # not i = 20
        # _id in [1,2,3]
        # _id not in [1,2,3]
        # a in ['a','b','c']
        # b not in ['a','b','c']
        # For now use term queries
        if field == Index.ID_INDEX_FIELD:
            task = ('query_ids', ('must_not', items))
        else:
            task = ('query_terms', ('must_not', (field, items)))
        self._tasks.append(task)

    @staticmethod
    def _sort_params_to_postprocessing(input):
        # Split string
        sort_params = input.split(":")

        query_sort_field = sort_params[0]
        query_sort_order = Operations.SortOrder.from_string(sort_params[1])

        task = ('sort_field', (query_sort_field, query_sort_order))

        return task

    @staticmethod
    def _query_params_to_size_and_sort(query_params):
        sort_params = None
        if query_params['query_sort_field'] and query_params['query_sort_order']:
            sort_params = query_params['query_sort_field'] + ":" + Operations.SortOrder.to_string(
                query_params['query_sort_order'])

        size = query_params['query_size']

        return size, sort_params

    @staticmethod
    def _count_post_processing(post_processing):
        size = None
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
                field_name_indexer = action[1][1]
                if index_indexer is None:
                    index_indexer = slice(None)
                if field_name_indexer is None:
                    field_name_indexer = slice(None)
                df = df.iloc[index_indexer, field_name_indexer]
            # field_names could be in here (and we ignore it)

        return df

    def _resolve_tasks(self):
        # We now try and combine all tasks into an Elasticsearch query
        # Some operations can be simply combined into a single query
        # other operations require pre-queries and then combinations
        # other operations require in-core post-processing of results
        query_params = {"query_sort_field": None,
                        "query_sort_order": None,
                        "query_size": None,
                        "query_fields": None,
                        "query_script_fields": None,
                        "query": Query()}

        post_processing = []

        for task in self._tasks:
            if task[0] == 'head':
                query_params, post_processing = self._resolve_head(task, query_params, post_processing)
            elif task[0] == 'tail':
                query_params, post_processing = self._resolve_tail(task, query_params, post_processing)
            elif task[0] == 'iloc':
                query_params, post_processing = self._resolve_iloc(task, query_params, post_processing)
            elif task[0] == 'query_ids':
                query_params, post_processing = self._resolve_query_ids(task, query_params, post_processing)
            elif task[0] == 'query_terms':
                query_params, post_processing = self._resolve_query_terms(task, query_params, post_processing)
            elif task[0] == 'boolean_filter':
                query_params, post_processing = self._resolve_boolean_filter(task, query_params, post_processing)
            elif task[0] == 'arithmetic_op_fields':
                query_params, post_processing = self._resolve_arithmetic_op_fields(task, query_params, post_processing)
            else:  # a lot of operations simply post-process the dataframe - put these straight through
                query_params, post_processing = self._resolve_post_processing_task(task, query_params, post_processing)

        return query_params, post_processing

    @staticmethod
    def _resolve_head(item, query_params, post_processing):
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

    @staticmethod
    def _resolve_tail(item, query_params, post_processing):
        # tail - sort desc, size n, post-process sort asc
        # |-------------12345|
        query_sort_field = item[1][0]
        query_sort_order = Operations.SortOrder.DESC
        query_size = item[1][1]

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
            post_processing.append(item)
            return query_params, post_processing

        # If results are already constrained, just get 'tail' of these
        # (note, currently we just append another tail, we don't optimise by
        # overwriting previous tail)
        if query_params['query_size'] is not None:
            post_processing.append(item)
            return query_params, post_processing
        else:
            query_params['query_size'] = query_size
        if query_params['query_sort_field'] is None:
            query_params['query_sort_field'] = query_sort_field
        if query_params['query_sort_order'] is None:
            query_params['query_sort_order'] = query_sort_order
        else:
            # reverse sort order
            query_params['query_sort_order'] = Operations.SortOrder.reverse(query_sort_order)

        post_processing.append('sort_index')

        return query_params, post_processing

    @staticmethod
    def _resolve_iloc(item, query_params, post_processing):
        # tail - sort desc, size n, post-process sort asc
        # |---4--7-9---------|

        # This is a list of items we return via an integer index
        int_index = item[1][0]
        if int_index is not None:
            last_item = int_index.max()

            # If we have a query_size we do this post processing
            if query_params['query_size'] is not None:
                post_processing.append(item)
                return query_params, post_processing

            # size should be > last item
            query_params['query_size'] = last_item + 1
        post_processing.append(item)

        return query_params, post_processing

    @staticmethod
    def _resolve_query_ids(item, query_params, post_processing):
        # task = ('query_ids', ('must_not', items))
        must_clause = item[1][0]
        ids = item[1][1]

        if must_clause == 'must':
            query_params['query'].ids(ids, must=True)
        else:
            query_params['query'].ids(ids, must=False)

        return query_params, post_processing

    @staticmethod
    def _resolve_query_terms(item, query_params, post_processing):
        # task = ('query_terms', ('must_not', (field, terms)))
        must_clause = item[1][0]
        field = item[1][1][0]
        terms = item[1][1][1]

        if must_clause == 'must':
            query_params['query'].terms(field, terms, must=True)
        else:
            query_params['query'].terms(field, terms, must=False)

        return query_params, post_processing

    @staticmethod
    def _resolve_boolean_filter(item, query_params, post_processing):
        # task = ('boolean_filter', object)
        boolean_filter = item[1]

        query_params['query'].update_boolean_filter(boolean_filter)

        return query_params, post_processing

    def _resolve_arithmetic_op_fields(self, item, query_params, post_processing):
        # task = ('arithmetic_op_fields', (field_name, (op_name, (left_field, right_field))))
        field_name = item[1][0]
        op_name = item[1][1][0]
        left_field = item[1][1][1][0]
        right_field = item[1][1][1][1]

        try:
            op_type = item[2]
        except IndexError:
            op_type = None

        # https://www.elastic.co/guide/en/elasticsearch/painless/current/painless-api-reference-shared-java-lang.html#painless-api-reference-shared-Math
        if not op_type:
            if isinstance(left_field, str) and isinstance(right_field, str):
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
                if op_name == '__add__':
                    source = "doc['{0}'].value + doc['{1}'].value".format(left_field, right_field)
                elif op_name == '__truediv__':
                    source = "doc['{0}'].value / doc['{1}'].value".format(left_field, right_field)
                elif op_name == '__floordiv__':
                    source = "Math.floor(doc['{0}'].value / doc['{1}'].value)".format(left_field, right_field)
                elif op_name == '__pow__':
                    source = "Math.pow(doc['{0}'].value, doc['{1}'].value)".format(left_field, right_field)
                elif op_name == '__mod__':
                    source = "doc['{0}'].value % doc['{1}'].value".format(left_field, right_field)
                elif op_name == '__mul__':
                    source = "doc['{0}'].value * doc['{1}'].value".format(left_field, right_field)
                elif op_name == '__sub__':
                    source = "doc['{0}'].value - doc['{1}'].value".format(left_field, right_field)
                else:
                    raise NotImplementedError("Not implemented operation '{0}'".format(op_name))

                if query_params['query_script_fields'] is None:
                    query_params['query_script_fields'] = {}
                query_params['query_script_fields'][field_name] = {
                    'script': {
                        'source': source
                    }
                }
            elif isinstance(left_field, str) and np.issubdtype(np.dtype(type(right_field)), np.number):
                """
                (if op_name = '__truediv__')

                "script_fields": {
                    "field_name": {
                    "script": {
                        "source": "doc[left_field].value / right_field"
                    }
                    }
                }
                """
                if op_name == '__add__':
                    source = "doc['{0}'].value + {1}".format(left_field, right_field)
                elif op_name == '__truediv__':
                    source = "doc['{0}'].value / {1}".format(left_field, right_field)
                elif op_name == '__floordiv__':
                    source = "Math.floor(doc['{0}'].value / {1})".format(left_field, right_field)
                elif op_name == '__pow__':
                    source = "Math.pow(doc['{0}'].value, {1})".format(left_field, right_field)
                elif op_name == '__mod__':
                    source = "doc['{0}'].value % {1}".format(left_field, right_field)
                elif op_name == '__mul__':
                    source = "doc['{0}'].value * {1}".format(left_field, right_field)
                elif op_name == '__sub__':
                    source = "doc['{0}'].value - {1}".format(left_field, right_field)
                else:
                    raise NotImplementedError("Not implemented operation '{0}'".format(op_name))
            elif np.issubdtype(np.dtype(type(left_field)), np.number) and isinstance(right_field, str):
                """
                (if op_name = '__truediv__')

                "script_fields": {
                    "field_name": {
                    "script": {
                        "source": "left_field / doc['right_field'].value"
                    }
                    }
                }
                """
                if op_name == '__add__':
                    source = "{0} + doc['{1}'].value".format(left_field, right_field)
                elif op_name == '__truediv__':
                    source = "{0} / doc['{1}'].value".format(left_field, right_field)
                elif op_name == '__floordiv__':
                    source = "Math.floor({0} / doc['{1}'].value)".format(left_field, right_field)
                elif op_name == '__pow__':
                    source = "Math.pow({0}, doc['{1}'].value)".format(left_field, right_field)
                elif op_name == '__mod__':
                    source = "{0} % doc['{1}'].value".format(left_field, right_field)
                elif op_name == '__mul__':
                    source = "{0} * doc['{1}'].value".format(left_field, right_field)
                elif op_name == '__sub__':
                    source = "{0} - doc['{1}'].value".format(left_field, right_field)
                else:
                    raise NotImplementedError("Not implemented operation '{0}'".format(op_name))

            else:
                raise TypeError("Types for operation inconsistent {} {} {}", type(left_field), type(right_field), op_name)

        elif op_type[0] == "string":
            # we need to check the type of string addition
            if op_type[1] == "s":
                """
                (if op_name = '__add__')

                "script_fields": {
                    "field_name": {
                    "script": {
                        "source": "doc[left_field].value + doc[right_field].value"
                    }
                    }
                }
                """
                if op_name == '__add__':
                    source = "doc['{0}'].value + doc['{1}'].value".format(left_field, right_field)
                else:
                    raise NotImplementedError("Not implemented operation '{0}'".format(op_name))

            elif op_type[1] == "r":
                if isinstance(left_field, str) and isinstance(right_field, str):
                    """
                    (if op_name = '__add__')

                    "script_fields": {
                        "field_name": {
                        "script": {
                            "source": "doc[left_field].value + right_field"
                        }
                        }
                    }
                    """
                    if op_name == '__add__':
                        source = "doc['{0}'].value + '{1}'".format(left_field, right_field)
                    else:
                        raise NotImplementedError("Not implemented operation '{0}'".format(op_name))

            elif op_type[1] == 'l':
                if isinstance(left_field, str) and isinstance(right_field, str):
                    """
                    (if op_name = '__add__')

                    "script_fields": {
                        "field_name": {
                        "script": {
                            "source": "left_field + doc[right_field].value"
                        }
                        }
                    }
                    """
                    if op_name == '__add__':
                        source = "'{0}' + doc['{1}'].value".format(left_field, right_field)
                    else:
                        raise NotImplementedError("Not implemented operation '{0}'".format(op_name))

        if query_params['query_script_fields'] is None:
            query_params['query_script_fields'] = {}
        query_params['query_script_fields'][field_name] = {
            'script': {
                'source': source
            }
        }

        return query_params, post_processing

    @staticmethod
    def _resolve_post_processing_task(item, query_params, post_processing):
        # Just do this in post-processing
        if item[0] != 'field_names':
            post_processing.append(item)

        return query_params, post_processing

    def _size(self, query_params, post_processing):
        # Shrink wrap code around checking if size parameter is set
        size = query_params['query_size']  # can be None

        pp_size = self._count_post_processing(post_processing)
        if pp_size is not None:
            if size is not None:
                size = min(size, pp_size)
            else:
                size = pp_size

        # This can return None
        return size

    def info_es(self, buf):
        buf.write("Operations:\n")
        buf.write(" tasks: {0}\n".format(self._tasks))

        query_params, post_processing = self._resolve_tasks()
        size, sort_params = Operations._query_params_to_size_and_sort(query_params)
        field_names = self.get_field_names()

        script_fields = query_params['query_script_fields']
        query = Query(query_params['query'])
        body = query.to_search_body()
        if script_fields is not None:
            body['script_fields'] = script_fields

        buf.write(" size: {0}\n".format(size))
        buf.write(" sort_params: {0}\n".format(sort_params))
        buf.write(" _source: {0}\n".format(field_names))
        buf.write(" body: {0}\n".format(body))
        buf.write(" post_processing: {0}\n".format(post_processing))

    def update_query(self, boolean_filter):
        task = ('boolean_filter', boolean_filter)
        self._tasks.append(task)
