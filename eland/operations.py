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
import copy
import warnings
from collections import OrderedDict

import pandas as pd
from pandas.core.dtypes.common import is_datetime_or_timedelta_dtype

from eland import Index, SortOrder, DEFAULT_CSV_BATCH_OUTPUT_SIZE, DEFAULT_ES_MAX_RESULT_WINDOW, \
    elasticsearch_date_to_pandas_date
from eland import Query
from eland.actions import SortFieldAction
from eland.tasks import HeadTask, TailTask, BooleanFilterTask, ArithmeticOpFieldsTask, QueryTermsTask, \
    QueryIdsTask, SizeTask


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

    def __init__(self, tasks=None, arithmetic_op_fields_task=None):
        if tasks is None:
            self._tasks = []
        else:
            self._tasks = tasks
        self._arithmetic_op_fields_task = arithmetic_op_fields_task

    def __constructor__(self, *args, **kwargs):
        return type(self)(*args, **kwargs)

    def copy(self):
        return self.__constructor__(tasks=copy.deepcopy(self._tasks),
                                    arithmetic_op_fields_task=copy.deepcopy(self._arithmetic_op_fields_task))

    def head(self, index, n):
        # Add a task that is an ascending sort with size=n
        task = HeadTask(index.sort_field, n)
        self._tasks.append(task)

    def tail(self, index, n):
        # Add a task that is descending sort with size=n
        task = TailTask(index.sort_field, n)
        self._tasks.append(task)

    def arithmetic_op_fields(self, display_name, arithmetic_series):
        if self._arithmetic_op_fields_task is None:
            self._arithmetic_op_fields_task = ArithmeticOpFieldsTask(display_name, arithmetic_series)
        else:
            self._arithmetic_op_fields_task.update(display_name, arithmetic_series)

    def get_arithmetic_op_fields(self):
        # get an ArithmeticOpFieldsTask if it exists
        return self._arithmetic_op_fields_task

    def __repr__(self):
        return repr(self._tasks)

    def count(self, query_compiler):
        query_params, post_processing = self._resolve_tasks(query_compiler)

        # Elasticsearch _count is very efficient and so used to return results here. This means that
        # data frames that have restricted size or sort params will not return valid results
        # (_count doesn't support size).
        # Longer term we may fall back to pandas, but this may result in loading all index into memory.
        if self._size(query_params, post_processing) is not None:
            raise NotImplementedError("Requesting count with additional query and processing parameters "
                                      "not supported {0} {1}"
                                      .format(query_params, post_processing))

        # Only return requested field_names
        fields = query_compiler.get_field_names(include_scripted_fields=False)

        counts = OrderedDict()
        for field in fields:
            body = Query(query_params['query'])
            body.exists(field, must=True)

            field_exists_count = query_compiler._client.count(index=query_compiler._index_pattern,
                                                              body=body.to_count_body())
            counts[field] = field_exists_count

        return pd.Series(data=counts, index=fields)

    def mean(self, query_compiler, numeric_only=True):
        return self._metric_aggs(query_compiler, 'avg', numeric_only=numeric_only)

    def sum(self, query_compiler, numeric_only=True):
        return self._metric_aggs(query_compiler, 'sum', numeric_only=numeric_only)

    def max(self, query_compiler, numeric_only=True):
        return self._metric_aggs(query_compiler, 'max', numeric_only=numeric_only)

    def min(self, query_compiler, numeric_only=True):
        return self._metric_aggs(query_compiler, 'min', numeric_only=numeric_only)

    def nunique(self, query_compiler):
        return self._metric_aggs(query_compiler, 'cardinality', field_types='aggregatable')

    def value_counts(self, query_compiler, es_size):
        return self._terms_aggs(query_compiler, 'terms', es_size)

    def hist(self, query_compiler, bins):
        return self._hist_aggs(query_compiler, bins)

    def _metric_aggs(self, query_compiler, func, field_types=None, numeric_only=None):
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
        query_params, post_processing = self._resolve_tasks(query_compiler)

        size = self._size(query_params, post_processing)
        if size is not None:
            raise NotImplementedError("Can not count field matches if size is set {}".format(size))

        body = Query(query_params['query'])

        results = OrderedDict()

        # some metrics aggs (including cardinality) work on all aggregatable fields
        # therefore we include an optional all parameter on operations
        # that call _metric_aggs
        if field_types == 'aggregatable':
            aggregatable_field_names = query_compiler._mappings.aggregatable_field_names()

            for field in aggregatable_field_names.keys():
                body.metric_aggs(field, func, field)

            response = query_compiler._client.search(
                index=query_compiler._index_pattern,
                size=0,
                body=body.to_search_body())

            # Results are of the form
            # "aggregations" : {
            #   "customer_full_name.keyword" : {
            #     "value" : 10
            #   }
            # }

            # map aggregatable (e.g. x.keyword) to field_name
            for key, value in aggregatable_field_names.items():
                results[value] = response['aggregations'][key]['value']
        else:
            if numeric_only:
                pd_dtypes, source_fields, date_formats = query_compiler._mappings.metric_source_fields(
                    include_bool=True)
            else:
                # The only non-numerics we support are bool and timestamps currently
                # strings are not supported by metric aggs in ES
                # TODO - sum isn't supported for Timestamp in pandas - although ES does attempt to do it
                pd_dtypes, source_fields, date_formats = query_compiler._mappings.metric_source_fields(
                    include_bool=True,
                    include_timestamp=True)

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
            #   },
            #   "timestamp": {
            #     "value": 1.5165624455644382E12,
            #     "value_as_string": "2018-01-21T19:20:45.564Z"
            #   }
            # }
            for pd_dtype, field, date_format in zip(pd_dtypes, source_fields, date_formats):
                if is_datetime_or_timedelta_dtype(pd_dtype):
                    results[field] = elasticsearch_date_to_pandas_date(
                        response['aggregations'][field]['value_as_string'],
                        date_format
                    )
                else:
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
        query_params, post_processing = self._resolve_tasks(query_compiler)

        size = self._size(query_params, post_processing)
        if size is not None:
            raise NotImplementedError("Can not count field matches if size is set {}".format(size))

        # Get just aggregatable field_names
        aggregatable_field_names = query_compiler._mappings.aggregatable_field_names()

        body = Query(query_params['query'])

        for field in aggregatable_field_names.keys():
            body.terms_aggs(field, func, field, es_size=es_size)

        response = query_compiler._client.search(
            index=query_compiler._index_pattern,
            size=0,
            body=body.to_search_body())

        results = OrderedDict()

        for key in aggregatable_field_names.keys():
            # key is aggregatable field, value is label
            # e.g. key=category.keyword, value=category
            for bucket in response['aggregations'][key]['buckets']:
                results[bucket['key']] = bucket['doc_count']

        try:
            # get first value in dict (key is .keyword)
            name = list(aggregatable_field_names.values())[0]
        except IndexError:
            name = None

        s = pd.Series(data=results, index=results.keys(), name=name)

        return s

    def _hist_aggs(self, query_compiler, num_bins):
        # Get histogram bins and weights for numeric field_names
        query_params, post_processing = self._resolve_tasks(query_compiler)

        size = self._size(query_params, post_processing)
        if size is not None:
            raise NotImplementedError("Can not count field matches if size is set {}".format(size))

        numeric_source_fields = query_compiler._mappings.numeric_source_fields()

        body = Query(query_params['query'])

        min_aggs = self._metric_aggs(query_compiler, 'min', numeric_only=True)
        max_aggs = self._metric_aggs(query_compiler, 'max', numeric_only=True)

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

        bins = OrderedDict()
        weights = OrderedDict()

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

            # in case of series let plotting.ed_hist_series thrown an exception
            if not response.get('aggregations'):
                continue

            # in case of dataframe, throw warning that field is excluded
            if not response['aggregations'].get(field):
                warnings.warn("{} has no meaningful histogram interval and will be excluded. "
                              "All values 0."
                              .format(field),
                              UserWarning)
                continue

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
        query_params, post_processing = self._resolve_tasks(query_compiler)

        size = self._size(query_params, post_processing)
        if size is not None:
            raise NotImplementedError("Can not count field matches if size is set {}".format(size))

        field_names = query_compiler.get_field_names(include_scripted_fields=False)

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
        results = OrderedDict()

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
        query_params, post_processing = self._resolve_tasks(query_compiler)

        size = self._size(query_params, post_processing)
        if size is not None:
            raise NotImplementedError("Can not count field matches if size is set {}".format(size))

        numeric_source_fields = query_compiler._mappings.numeric_source_fields()

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

        results = OrderedDict()

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

    def to_pandas(self, query_compiler, show_progress=False):
        class PandasDataFrameCollector:
            def __init__(self, show_progress):
                self._df = None
                self._show_progress = show_progress

            def collect(self, df):
                # This collector does not batch data on output. Therefore, batch_size is fixed to None and this method
                # is only called once.
                if self._df is not None:
                    raise RuntimeError("Logic error in execution, this method must only be called once for this"
                                       "collector - batch_size == None")
                self._df = df

            @staticmethod
            def batch_size():
                # Do not change (see notes on collect)
                return None

            @property
            def show_progress(self):
                return self._show_progress

        collector = PandasDataFrameCollector(show_progress)

        self._es_results(query_compiler, collector)

        return collector._df

    def to_csv(self, query_compiler, show_progress=False, **kwargs):
        class PandasToCSVCollector:
            def __init__(self, show_progress, **args):
                self._args = args
                self._show_progress = show_progress
                self._ret = None
                self._first_time = True

            def collect(self, df):
                # If this is the first time we collect results, then write header, otherwise don't write header
                # and append results
                if self._first_time:
                    self._first_time = False
                    df.to_csv(**self._args)
                else:
                    # Don't write header, and change mode to append
                    self._args['header'] = False
                    self._args['mode'] = 'a'
                    df.to_csv(**self._args)

            @staticmethod
            def batch_size():
                # By default read n docs and then dump to csv
                batch_size = DEFAULT_CSV_BATCH_OUTPUT_SIZE
                return batch_size

            @property
            def show_progress(self):
                return self._show_progress

        collector = PandasToCSVCollector(show_progress, **kwargs)

        self._es_results(query_compiler, collector)

        return collector._ret

    def _es_results(self, query_compiler, collector):
        query_params, post_processing = self._resolve_tasks(query_compiler)

        size, sort_params = Operations._query_params_to_size_and_sort(query_params)

        script_fields = query_params['query_script_fields']
        query = Query(query_params['query'])

        body = query.to_search_body()
        if script_fields is not None:
            body['script_fields'] = script_fields

        # Only return requested field_names
        _source = query_compiler.get_field_names(include_scripted_fields=False)
        if _source:
            # For query_compiler._client.search we could add _source
            # as a parameter, or add this value in body.
            #
            # If _source is a parameter it is encoded into to the url.
            #
            # If _source is a large number of fields (1000+) then this can result in an
            # extremely long url and a `too_long_frame_exception`. Therefore, add
            # _source to the body rather than as a _source parameter
            body['_source'] = _source
        else:
            _source = False

        es_results = None

        # If size=None use scan not search - then post sort results when in df
        # If size>10000 use scan
        is_scan = False
        if size is not None and size <= DEFAULT_ES_MAX_RESULT_WINDOW:
            if size > 0:
                try:


                    es_results = query_compiler._client.search(
                        index=query_compiler._index_pattern,
                        size=size,
                        sort=sort_params,
                        body=body)
                except Exception:
                    # Catch all ES errors and print debug (currently to stdout)
                    error = {
                        'index': query_compiler._index_pattern,
                        'size': size,
                        'sort': sort_params,
                        'body': body
                    }
                    print("Elasticsearch error:", error)
                    raise
        else:
            is_scan = True
            es_results = query_compiler._client.scan(
                index=query_compiler._index_pattern,
                query=body)
            # create post sort
            if sort_params is not None:
                post_processing.append(SortFieldAction(sort_params))

        if is_scan:
            while True:
                partial_result, df = query_compiler._es_results_to_pandas(es_results, collector.batch_size(),
                                                                          collector.show_progress)
                df = self._apply_df_post_processing(df, post_processing)
                collector.collect(df)
                if not partial_result:
                    break
        else:
            partial_result, df = query_compiler._es_results_to_pandas(es_results)
            df = self._apply_df_post_processing(df, post_processing)
            collector.collect(df)

    def index_count(self, query_compiler, field):
        # field is the index field so count values
        query_params, post_processing = self._resolve_tasks(query_compiler)

        size = self._size(query_params, post_processing)

        # Size is dictated by operations
        if size is not None:
            # TODO - this is not necessarily valid as the field may not exist in ALL these docs
            return size

        body = Query(query_params['query'])
        body.exists(field, must=True)

        return query_compiler._client.count(index=query_compiler._index_pattern, body=body.to_count_body())

    def _validate_index_operation(self, query_compiler, items):
        if not isinstance(items, list):
            raise TypeError("list item required - not {}".format(type(items)))

        # field is the index field so count values
        query_params, post_processing = self._resolve_tasks(query_compiler)

        size = self._size(query_params, post_processing)

        # Size is dictated by operations
        if size is not None:
            raise NotImplementedError("Can not count field matches if size is set {}".format(size))

        return query_params, post_processing

    def index_matches_count(self, query_compiler, field, items):
        query_params, post_processing = self._validate_index_operation(query_compiler, items)

        body = Query(query_params['query'])

        if field == Index.ID_INDEX_FIELD:
            body.ids(items, must=True)
        else:
            body.terms(field, items, must=True)

        return query_compiler._client.count(index=query_compiler._index_pattern, body=body.to_count_body())

    def drop_index_values(self, query_compiler, field, items):
        self._validate_index_operation(query_compiler, items)

        # Putting boolean queries together
        # i = 10
        # not i = 20
        # _id in [1,2,3]
        # _id not in [1,2,3]
        # a in ['a','b','c']
        # b not in ['a','b','c']
        # For now use term queries
        if field == Index.ID_INDEX_FIELD:
            task = QueryIdsTask(False, items)
        else:
            task = QueryTermsTask(False, field, items)
        self._tasks.append(task)

    @staticmethod
    def _query_params_to_size_and_sort(query_params):
        sort_params = None
        if query_params['query_sort_field'] and query_params['query_sort_order']:
            sort_params = query_params['query_sort_field'] + ":" + SortOrder.to_string(
                query_params['query_sort_order'])

        size = query_params['query_size']

        return size, sort_params

    @staticmethod
    def _count_post_processing(post_processing):
        size = None
        for action in post_processing:
            if isinstance(action, SizeTask):
                if size is None or action.size() < size:
                    size = action.size()

        return size

    @staticmethod
    def _apply_df_post_processing(df, post_processing):
        for action in post_processing:
            df = action.resolve_action(df)

        return df

    def _resolve_tasks(self, query_compiler):
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
            query_params, post_processing = task.resolve_task(query_params, post_processing, query_compiler)

        if self._arithmetic_op_fields_task is not None:
            query_params, post_processing = self._arithmetic_op_fields_task.resolve_task(query_params, post_processing,
                                                                                         query_compiler)

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

    def info_es(self, query_compiler, buf):
        buf.write("Operations:\n")
        buf.write(" tasks: {0}\n".format(self._tasks))

        query_params, post_processing = self._resolve_tasks(query_compiler)
        size, sort_params = Operations._query_params_to_size_and_sort(query_params)
        _source = query_compiler._mappings.get_field_names()

        script_fields = query_params['query_script_fields']
        query = Query(query_params['query'])
        body = query.to_search_body()
        if script_fields is not None:
            body['script_fields'] = script_fields

        buf.write(" size: {0}\n".format(size))
        buf.write(" sort_params: {0}\n".format(sort_params))
        buf.write(" _source: {0}\n".format(_source))
        buf.write(" body: {0}\n".format(body))
        buf.write(" post_processing: {0}\n".format(post_processing))

    def update_query(self, boolean_filter):
        task = BooleanFilterTask(boolean_filter)
        self._tasks.append(task)
