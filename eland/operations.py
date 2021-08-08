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

import copy
import warnings
from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    TextIO,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd  # type: ignore
from elasticsearch.exceptions import NotFoundError

from eland.actions import PostProcessingAction
from eland.common import (
    DEFAULT_CSV_BATCH_OUTPUT_SIZE,
    DEFAULT_PAGINATION_SIZE,
    DEFAULT_PIT_KEEP_ALIVE,
    DEFAULT_SEARCH_SIZE,
    SortOrder,
    build_pd_series,
    elasticsearch_date_to_pandas_date,
    es_version,
)
from eland.index import Index
from eland.query import Query
from eland.tasks import (
    RESOLVED_TASK_TYPE,
    ArithmeticOpFieldsTask,
    BooleanFilterTask,
    HeadTask,
    QueryIdsTask,
    QueryTermsTask,
    SampleTask,
    SizeTask,
    TailTask,
)

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

    from eland.arithmetics import ArithmeticSeries
    from eland.field_mappings import Field
    from eland.filter import BooleanFilter
    from eland.query_compiler import QueryCompiler
    from eland.tasks import Task


class QueryParams:
    def __init__(self) -> None:
        self.query: Query = Query()
        self.sort_field: Optional[str] = None
        self.sort_order: Optional[SortOrder] = None
        self.size: Optional[int] = None
        self.fields: Optional[List[str]] = None
        self.script_fields: Optional[Dict[str, Dict[str, Any]]] = None


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

    def __init__(
        self,
        tasks: Optional[List["Task"]] = None,
        arithmetic_op_fields_task: Optional["ArithmeticOpFieldsTask"] = None,
    ) -> None:
        self._tasks: List["Task"]
        if tasks is None:
            self._tasks = []
        else:
            self._tasks = tasks
        self._arithmetic_op_fields_task = arithmetic_op_fields_task

    def __constructor__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> "Operations":
        return type(self)(*args, **kwargs)

    def copy(self) -> "Operations":
        return self.__constructor__(
            tasks=copy.deepcopy(self._tasks),
            arithmetic_op_fields_task=copy.deepcopy(self._arithmetic_op_fields_task),
        )

    def head(self, index: "Index", n: int) -> None:
        # Add a task that is an ascending sort with size=n
        task = HeadTask(index, n)
        self._tasks.append(task)

    def tail(self, index: "Index", n: int) -> None:
        # Add a task that is descending sort with size=n
        task = TailTask(index, n)
        self._tasks.append(task)

    def sample(self, index: "Index", n: int, random_state: int) -> None:
        task = SampleTask(index, n, random_state)
        self._tasks.append(task)

    def arithmetic_op_fields(
        self, display_name: str, arithmetic_series: "ArithmeticSeries"
    ) -> None:
        if self._arithmetic_op_fields_task is None:
            self._arithmetic_op_fields_task = ArithmeticOpFieldsTask(
                display_name, arithmetic_series
            )
        else:
            self._arithmetic_op_fields_task.update(display_name, arithmetic_series)

    def get_arithmetic_op_fields(self) -> Optional[ArithmeticOpFieldsTask]:
        # get an ArithmeticOpFieldsTask if it exists
        return self._arithmetic_op_fields_task

    def __repr__(self) -> str:
        return repr(self._tasks)

    def count(self, query_compiler: "QueryCompiler") -> pd.Series:
        query_params, post_processing = self._resolve_tasks(query_compiler)

        # Elasticsearch _count is very efficient and so used to return results here. This means that
        # data frames that have restricted size or sort params will not return valid results
        # (_count doesn't support size).
        # Longer term we may fall back to pandas, but this may result in loading all index into memory.
        if self._size(query_params, post_processing) is not None:
            raise NotImplementedError(
                f"Requesting count with additional query and processing parameters "
                f"not supported {query_params} {post_processing}"
            )

        # Only return requested field_names
        fields = query_compiler.get_field_names(include_scripted_fields=False)

        counts = {}
        for field in fields:
            body = Query(query_params.query)
            body.exists(field, must=True)

            field_exists_count = query_compiler._client.count(
                index=query_compiler._index_pattern, body=body.to_count_body()
            )["count"]
            counts[field] = field_exists_count

        return build_pd_series(data=counts, index=fields)

    def _metric_agg_series(
        self,
        query_compiler: "QueryCompiler",
        agg: List["str"],
        numeric_only: Optional[bool] = None,
    ) -> pd.Series:
        results = self._metric_aggs(query_compiler, agg, numeric_only=numeric_only)
        if numeric_only:
            return build_pd_series(results, index=results.keys(), dtype=np.float64)
        else:
            # If all results are float convert into float64
            if all(isinstance(i, float) for i in results.values()):
                dtype: "DTypeLike" = np.float64
            # If all results are int convert into int64
            elif all(isinstance(i, int) for i in results.values()):
                dtype = np.int64
            # If single result is present consider that datatype instead of object
            elif len(results) <= 1:
                dtype = None
            else:
                dtype = "object"
            return build_pd_series(results, index=results.keys(), dtype=dtype)

    def value_counts(self, query_compiler: "QueryCompiler", es_size: int) -> pd.Series:
        return self._terms_aggs(query_compiler, "terms", es_size)

    def hist(
        self, query_compiler: "QueryCompiler", bins: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self._hist_aggs(query_compiler, bins)

    def idx(
        self, query_compiler: "QueryCompiler", axis: int, sort_order: str
    ) -> pd.Series:

        if axis == 1:
            # Fetch idx on Columns
            raise NotImplementedError(
                "This feature is not implemented yet for 'axis = 1'"
            )

        # Fetch idx on Index
        query_params, post_processing = self._resolve_tasks(query_compiler)

        fields = query_compiler._mappings.all_source_fields()

        # Consider only Numeric fields
        fields = [field for field in fields if (field.is_numeric)]

        body = Query(query_params.query)

        for field in fields:
            body.top_hits_agg(
                name=f"top_hits_{field.es_field_name}",
                source_columns=[field.es_field_name],
                sort_order=sort_order,
                size=1,
            )

        # Fetch Response
        response = query_compiler._client.search(
            index=query_compiler._index_pattern, size=0, body=body.to_search_body()
        )
        response = response["aggregations"]

        results = {}
        for field in fields:
            res = response[f"top_hits_{field.es_field_name}"]["hits"]

            if not res["total"]["value"] > 0:
                raise ValueError("Empty Index with no rows")

            if not res["hits"][0]["_source"]:
                # This means there are NaN Values, we skip them
                # Implement this when skipna is implemented
                continue
            else:
                results[field.es_field_name] = res["hits"][0]["_id"]

        return pd.Series(results)

    def aggs(
        self,
        query_compiler: "QueryCompiler",
        pd_aggs: List[str],
        numeric_only: Optional[bool] = None,
    ) -> pd.DataFrame:
        results = self._metric_aggs(
            query_compiler, pd_aggs, numeric_only=numeric_only, is_dataframe_agg=True
        )
        return pd.DataFrame(
            results, index=pd_aggs, dtype=(np.float64 if numeric_only else None)
        )

    def mode(
        self,
        query_compiler: "QueryCompiler",
        pd_aggs: List[str],
        is_dataframe: bool,
        es_size: int,
        numeric_only: bool = False,
        dropna: bool = True,
    ) -> Union[pd.DataFrame, pd.Series]:

        results = self._metric_aggs(
            query_compiler,
            pd_aggs=pd_aggs,
            numeric_only=numeric_only,
            dropna=dropna,
            es_mode_size=es_size,
        )

        pd_dict: Dict[str, Any] = {}
        row_diff: Optional[int] = None

        if is_dataframe:
            # If multiple values of mode is returned for a particular column
            # find the maximum length and use that to fill dataframe with NaN/NaT
            rows_len = max(len(value) for value in results.values())
            for key, values in results.items():
                row_diff = rows_len - len(values)
                # Convert np.ndarray to list
                values = list(values)
                if row_diff:
                    if isinstance(values[0], pd.Timestamp):
                        values.extend([pd.NaT] * row_diff)
                    else:
                        values.extend([np.NaN] * row_diff)
                pd_dict[key] = values

            return pd.DataFrame(pd_dict)
        else:
            return pd.DataFrame(results.values()).iloc[0].rename()

    def _metric_aggs(
        self,
        query_compiler: "QueryCompiler",
        pd_aggs: List[str],
        numeric_only: Optional[bool] = None,
        is_dataframe_agg: bool = False,
        es_mode_size: Optional[int] = None,
        dropna: bool = True,
        percentiles: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Used to calculate metric aggregations
        https://www.elastic.co/guide/en/elasticsearch/reference/current/search-aggregations-metrics.html

        Parameters
        ----------
        query_compiler:
            Query Compiler object
        pd_aggs:
            aggregations that are to be performed on dataframe or series
        numeric_only:
            return either all numeric values or NaN/NaT
        is_dataframe_agg:
            know if this method is called from single-agg or aggreagation method
        es_mode_size:
            number of rows to return when multiple mode values are present.
        dropna:
            drop NaN/NaT for a dataframe
        percentiles:
            List of percentiles when 'quantile' agg is called. Otherwise it is None

        Returns
        -------
            A dictionary which contains all aggregations calculated.
        """
        query_params, post_processing = self._resolve_tasks(query_compiler)

        size = self._size(query_params, post_processing)
        if size is not None:
            raise NotImplementedError(
                f"Can not count field matches if size is set {size}"
            )

        fields = query_compiler._mappings.all_source_fields()
        if numeric_only:
            # Consider if field is Int/Float/Bool
            fields = [field for field in fields if (field.is_numeric or field.is_bool)]

        body = Query(query_params.query)

        # Convert pandas aggs to ES equivalent
        es_aggs = self._map_pd_aggs_to_es_aggs(pd_aggs, percentiles)

        for field in fields:
            for es_agg in es_aggs:
                # NaN/NaT fields are ignored
                if not field.is_es_agg_compatible(es_agg):
                    continue

                # If we have multiple 'extended_stats' etc. here we simply NOOP on 2nd call
                if isinstance(es_agg, tuple):
                    if es_agg[0] == "percentiles":
                        body.percentile_agg(
                            name=f"{es_agg[0]}_{field.es_field_name}",
                            field=field.es_field_name,
                            percents=es_agg[1],
                        )
                    else:
                        body.metric_aggs(
                            name=f"{es_agg[0]}_{field.es_field_name}",
                            func=es_agg[0],
                            field=field.aggregatable_es_field_name,
                        )
                elif es_agg == "mode":
                    # TODO for dropna=False, Check If field is timestamp or boolean or numeric,
                    # then use missing parameter for terms aggregation.
                    body.terms_aggs(
                        name=f"{es_agg}_{field.es_field_name}",
                        func="terms",
                        field=field.aggregatable_es_field_name,
                        es_size=es_mode_size,
                    )

                else:
                    body.metric_aggs(
                        name=f"{es_agg}_{field.es_field_name}",
                        func=es_agg,
                        field=field.aggregatable_es_field_name,
                    )

        response = query_compiler._client.search(
            index=query_compiler._index_pattern, size=0, body=body.to_search_body()
        )

        """
        Results are like (for 'sum', 'min')

             AvgTicketPrice  DistanceKilometers  DistanceMiles  FlightDelayMin
        sum    8.204365e+06        9.261629e+07   5.754909e+07          618150
        min    1.000205e+02        0.000000e+00   0.000000e+00               0
        """

        return self._unpack_metric_aggs(
            fields=fields,
            es_aggs=es_aggs,
            pd_aggs=pd_aggs,
            response=response,
            numeric_only=numeric_only,
            is_dataframe_agg=is_dataframe_agg,
            percentiles=percentiles,
        )

    def _terms_aggs(
        self, query_compiler: "QueryCompiler", func: str, es_size: int
    ) -> pd.Series:
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
            raise NotImplementedError(
                f"Can not count field matches if size is set {size}"
            )

        # Get just aggregatable field_names
        aggregatable_field_names = query_compiler._mappings.aggregatable_field_names()

        body = Query(query_params.query)

        for field in aggregatable_field_names.keys():
            body.terms_aggs(field, func, field, es_size=es_size)

        response = query_compiler._client.search(
            index=query_compiler._index_pattern, size=0, body=body.to_search_body()
        )

        results = {}

        for key in aggregatable_field_names.keys():
            # key is aggregatable field, value is label
            # e.g. key=category.keyword, value=category
            for bucket in response["aggregations"][key]["buckets"]:
                results[bucket["key"]] = bucket["doc_count"]

        try:
            # get first value in dict (key is .keyword)
            name: Optional[str] = list(aggregatable_field_names.values())[0]
        except IndexError:
            name = None

        return build_pd_series(results, name=name)

    def _hist_aggs(
        self, query_compiler: "QueryCompiler", num_bins: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Get histogram bins and weights for numeric field_names
        query_params, post_processing = self._resolve_tasks(query_compiler)

        size = self._size(query_params, post_processing)
        if size is not None:
            raise NotImplementedError(
                f"Can not count field matches if size is set {size}"
            )

        numeric_source_fields = query_compiler._mappings.numeric_source_fields()

        body = Query(query_params.query)

        results = self._metric_aggs(query_compiler, ["min", "max"], numeric_only=True)
        min_aggs = {}
        max_aggs = {}
        for field, (min_agg, max_agg) in results.items():
            min_aggs[field] = min_agg
            max_aggs[field] = max_agg

        for field in numeric_source_fields:
            body.hist_aggs(field, field, min_aggs[field], max_aggs[field], num_bins)

        response = query_compiler._client.search(
            index=query_compiler._index_pattern, size=0, body=body.to_search_body()
        )
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

        bins: Dict[str, List[int]] = {}
        weights: Dict[str, List[int]] = {}

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
            if not response.get("aggregations"):
                continue

            # in case of dataframe, throw warning that field is excluded
            if not response["aggregations"].get(field):
                warnings.warn(
                    f"{field} has no meaningful histogram interval and will be excluded. "
                    f"All values 0.",
                    UserWarning,
                )
                continue

            buckets = response["aggregations"][field]["buckets"]

            bins[field] = []
            weights[field] = []

            for bucket in buckets:
                bins[field].append(bucket["key"])

                if bucket == buckets[-1]:
                    weights[field][-1] += bucket["doc_count"]
                else:
                    weights[field].append(bucket["doc_count"])

        df_bins = pd.DataFrame(data=bins)
        df_weights = pd.DataFrame(data=weights)
        return df_bins, df_weights

    def _unpack_metric_aggs(
        self,
        fields: List["Field"],
        es_aggs: Union[List[str], List[Tuple[str, List[float]]]],
        pd_aggs: List[str],
        response: Dict[str, Any],
        numeric_only: Optional[bool],
        percentiles: Optional[Sequence[float]] = None,
        is_dataframe_agg: bool = False,
        is_groupby: bool = False,
    ) -> Dict[str, List[Any]]:
        """
        This method unpacks metric aggregations JSON response.
        This can be called either directly on an aggs query
        or on an individual bucket within a composite aggregation.

        Parameters
        ----------
        fields:
            a list of Field Mappings
        es_aggs:
            Eland Equivalent of aggs
        pd_aggs:
            a list of aggs
        response:
            a dict containing response from Elasticsearch
        numeric_only:
            return either numeric values or NaN/NaT
        is_dataframe_agg:
            - True then aggregation is called from dataframe
            - False then aggregation is called from series
        percentiles:
            List of percentiles when 'quantile' agg is called. Otherwise it is None

        Returns
        -------
            a dictionary on which agg caluculations are done.
        """
        results: Dict[str, Any] = {}
        percentile_values: List[float] = []
        agg_value: Any

        for field in fields:
            values = []
            for es_agg, pd_agg in zip(es_aggs, pd_aggs):
                # is_dataframe_agg is used to differentiate agg() and an aggregation called through .mean()
                # If the field and agg aren't compatible we add a NaN/NaT for agg
                # If the field and agg aren't compatible we don't add NaN/NaT for an aggregation called through .mean()
                if not field.is_es_agg_compatible(es_agg):
                    if is_dataframe_agg and not numeric_only:
                        values.append(field.nan_value)
                    elif not is_dataframe_agg and numeric_only is False:
                        values.append(field.nan_value)
                    # Explicit condition for mad to add NaN because it doesn't support bool
                    elif is_dataframe_agg and numeric_only:
                        if pd_agg == "mad":
                            values.append(field.nan_value)
                    continue

                if isinstance(es_agg, tuple):
                    agg_value = response["aggregations"][
                        f"{es_agg[0]}_{field.es_field_name}"
                    ]

                    # Pull multiple values from 'percentiles' result.
                    if es_agg[0] == "percentiles":
                        agg_value = agg_value["values"]  # Returns dictionary
                        if pd_agg == "median":
                            agg_value = agg_value["50.0"]
                        # Currently Pandas does the same
                        # If we call quantile it returns the same result as of median.
                        elif (
                            pd_agg == "quantile" and is_dataframe_agg and not is_groupby
                        ):
                            agg_value = agg_value["50.0"]
                        else:
                            # Maintain order of percentiles
                            if percentiles:
                                percentile_values = [
                                    agg_value[str(i)] for i in percentiles
                                ]

                    if not percentile_values and pd_agg not in ("quantile", "median"):
                        agg_value = agg_value[es_agg[1]]
                    # Need to convert 'Population' stddev and variance
                    # from Elasticsearch into 'Sample' stddev and variance
                    # which is what pandas uses.
                    if es_agg[1] in ("std_deviation", "variance"):
                        # Neither transformation works with count <=1
                        count = response["aggregations"][
                            f"{es_agg[0]}_{field.es_field_name}"
                        ]["count"]

                        # All of the below calculations result in NaN if count<=1
                        if count <= 1:
                            agg_value = np.NaN

                        elif es_agg[1] == "std_deviation":
                            agg_value *= count / (count - 1.0)

                        else:  # es_agg[1] == "variance"
                            # sample_std=\sqrt{\frac{1}{N-1}\sum_{i=1}^N(x_i-\bar{x})^2}
                            # population_std=\sqrt{\frac{1}{N}\sum_{i=1}^N(x_i-\bar{x})^2}
                            # sample_std=\sqrt{\frac{N}{N-1}population_std}
                            agg_value = np.sqrt(
                                (count / (count - 1.0)) * agg_value * agg_value
                            )
                elif es_agg == "mode":
                    # For terms aggregation buckets are returned
                    # agg_value will be of type list
                    agg_value = response["aggregations"][
                        f"{es_agg}_{field.es_field_name}"
                    ]["buckets"]
                else:
                    agg_value = response["aggregations"][
                        f"{es_agg}_{field.es_field_name}"
                    ]["value"]

                if isinstance(agg_value, list):
                    # include top-terms in the result.
                    if not agg_value:
                        # If the all the documents for a field are empty
                        agg_value = [field.nan_value]
                    else:
                        max_doc_count = agg_value[0]["doc_count"]
                        # We need only keys which are equal to max_doc_count
                        # lesser values are ignored
                        agg_value = [
                            item["key"]
                            for item in agg_value
                            if item["doc_count"] == max_doc_count
                        ]

                        # Maintain datatype by default because pandas does the same
                        # text are returned as-is
                        if field.is_bool or field.is_numeric:
                            agg_value = [
                                field.np_dtype.type(value) for value in agg_value
                            ]

                # Null usually means there were no results.
                if not isinstance(agg_value, (list, dict)) and (
                    agg_value is None or np.isnan(agg_value)
                ):
                    if is_dataframe_agg and not numeric_only:
                        agg_value = np.NaN
                    elif not is_dataframe_agg and numeric_only is False:
                        agg_value = np.NaN

                # Cardinality is always either NaN or integer.
                elif pd_agg in ("nunique", "count"):
                    agg_value = (
                        int(agg_value)
                        if isinstance(agg_value, (int, float))
                        else np.NaN
                    )

                # If this is a non-null timestamp field convert to a pd.Timestamp()
                elif field.is_timestamp:
                    if isinstance(agg_value, list):
                        # convert to timestamp results for mode
                        agg_value = [
                            elasticsearch_date_to_pandas_date(
                                value, field.es_date_format
                            )
                            for value in agg_value
                        ]
                    elif percentile_values:
                        percentile_values = [
                            elasticsearch_date_to_pandas_date(
                                value, field.es_date_format
                            )
                            for value in percentile_values
                        ]
                    else:
                        assert not isinstance(agg_value, dict)
                        agg_value = elasticsearch_date_to_pandas_date(
                            agg_value, field.es_date_format
                        )
                # If numeric_only is False | None then maintain column datatype
                elif not numeric_only and pd_agg != "quantile":
                    # we're only converting to bool for lossless aggs like min, max, and median.
                    if pd_agg in {"max", "min", "median", "sum", "mode"}:
                        # 'sum' isn't representable with bool, use int64
                        if pd_agg == "sum" and field.is_bool:
                            agg_value = np.int64(agg_value)
                        else:
                            agg_value = field.np_dtype.type(agg_value)

                if not percentile_values:
                    values.append(agg_value)

            # If numeric_only is True and We only have a NaN type field then we check for empty.
            if values:
                results[field.column] = values if len(values) > 1 else values[0]
            # This only runs when df.quantile() or series.quantile() or
            # quantile from groupby is called
            if percentile_values:
                results[f"{field.column}"] = percentile_values

        return results

    def quantile(
        self,
        query_compiler: "QueryCompiler",
        pd_aggs: List[str],
        quantiles: Union[int, float, List[int], List[float]],
        is_dataframe: bool = True,
        numeric_only: Optional[bool] = True,
    ) -> Union[pd.DataFrame, pd.Series]:

        percentiles = [
            quantile_to_percentile(x)
            for x in (
                (quantiles,) if not isinstance(quantiles, (list, tuple)) else quantiles
            )
        ]

        result = self._metric_aggs(
            query_compiler,
            pd_aggs=pd_aggs,
            percentiles=percentiles,
            is_dataframe_agg=False,
            numeric_only=numeric_only,
        )

        df = pd.DataFrame(
            result,
            index=[i / 100 for i in percentiles],
            columns=result.keys(),
            dtype=(np.float64 if numeric_only else None),
        )

        # Display Output same as pandas does
        if isinstance(quantiles, float):
            return df.squeeze()
        else:
            return df if is_dataframe else df.transpose().iloc[0]

    def aggs_groupby(
        self,
        query_compiler: "QueryCompiler",
        by: List[str],
        pd_aggs: List[str],
        dropna: bool = True,
        quantiles: Optional[Union[int, float, List[int], List[float]]] = None,
        is_dataframe_agg: bool = False,
        numeric_only: Optional[bool] = True,
    ) -> pd.DataFrame:
        """
        This method is used to construct groupby aggregation dataframe

        Parameters
        ----------
        query_compiler:
            A Query compiler
        by:
            a list of columns on which groupby operations have to be performed
        pd_aggs:
            a list of aggregations to be performed
        dropna:
            Drop None values if True.
            TODO Not yet implemented
        is_dataframe_agg:
            Know if groupby with aggregation or single agg is called.
        numeric_only:
            return either numeric values or NaN/NaT
        quantiles:
            List of quantiles when 'quantile' agg is called. Otherwise it is None

        Returns
        -------
            A dataframe which consists groupby data
        """
        query_params, post_processing = self._resolve_tasks(query_compiler)

        size = self._size(query_params, post_processing)
        if size is not None:
            raise NotImplementedError(
                f"Can not count field matches if size is set {size}"
            )

        by_fields, agg_fields = query_compiler._mappings.groupby_source_fields(by=by)

        # Used defaultdict to avoid initialization of columns with lists
        results: Dict[Any, List[Any]] = defaultdict(list)

        if numeric_only:
            agg_fields = [
                field for field in agg_fields if (field.is_numeric or field.is_bool)
            ]

        body = Query(query_params.query)

        # To return for creating multi-index on columns
        headers = [agg_field.column for agg_field in agg_fields]

        percentiles: Optional[List[float]] = None
        len_percentiles: int = 0
        if quantiles:
            percentiles = [
                quantile_to_percentile(x)
                for x in (
                    (quantiles,)
                    if not isinstance(quantiles, (list, tuple))
                    else quantiles
                )
            ]
            len_percentiles = len(percentiles)

        # Convert pandas aggs to ES equivalent
        es_aggs = self._map_pd_aggs_to_es_aggs(pd_aggs=pd_aggs, percentiles=percentiles)

        # Construct Query
        for by_field in by_fields:
            if by_field.aggregatable_es_field_name is None:
                raise ValueError(
                    f"Cannot use {by_field.column!r} with groupby() because "
                    f"it has no aggregatable fields in Elasticsearch"
                )
            # groupby fields will be term aggregations
            body.composite_agg_bucket_terms(
                name=f"groupby_{by_field.column}",
                field=by_field.aggregatable_es_field_name,
            )

        for agg_field in agg_fields:
            for es_agg in es_aggs:
                # Skip if the field isn't compatible or if the agg is
                # 'value_count' as this value is pulled from bucket.doc_count.
                if not agg_field.is_es_agg_compatible(es_agg):
                    continue

                # If we have multiple 'extended_stats' etc. here we simply NOOP on 2nd call
                if isinstance(es_agg, tuple):
                    if es_agg[0] == "percentiles":
                        body.percentile_agg(
                            name=f"{es_agg[0]}_{agg_field.es_field_name}",
                            field=agg_field.es_field_name,
                            percents=es_agg[1],
                        )
                    else:
                        body.metric_aggs(
                            f"{es_agg[0]}_{agg_field.es_field_name}",
                            es_agg[0],
                            agg_field.aggregatable_es_field_name,
                        )
                else:
                    body.metric_aggs(
                        f"{es_agg}_{agg_field.es_field_name}",
                        es_agg,
                        agg_field.aggregatable_es_field_name,
                    )

        # Composite aggregation
        body.composite_agg_start(
            size=DEFAULT_PAGINATION_SIZE, name="groupby_buckets", dropna=dropna
        )

        for buckets in self.bucket_generator(query_compiler, body):
            # We recieve response row-wise
            for bucket in buckets:
                # groupby columns are added to result same way they are returned
                for by_field in by_fields:
                    bucket_key = bucket["key"][f"groupby_{by_field.column}"]

                    # Datetimes always come back as integers, convert to pd.Timestamp()
                    if by_field.is_timestamp and isinstance(bucket_key, int):
                        bucket_key = pd.to_datetime(bucket_key, unit="ms")

                    if pd_aggs == ["quantile"] and len_percentiles > 1:
                        bucket_key = [bucket_key] * len_percentiles

                    results[by_field.column].extend(
                        bucket_key if isinstance(bucket_key, list) else [bucket_key]
                    )

                agg_calculation = self._unpack_metric_aggs(
                    fields=agg_fields,
                    es_aggs=es_aggs,
                    pd_aggs=pd_aggs,
                    response={"aggregations": bucket},
                    numeric_only=numeric_only,
                    percentiles=percentiles,
                    # We set 'True' here because we want the value
                    # unpacking to always be in 'dataframe' mode.
                    is_dataframe_agg=True,
                    is_groupby=True,
                )

                # to construct index with quantiles
                if pd_aggs == ["quantile"] and percentiles and len_percentiles > 1:
                    results[None].extend([i / 100 for i in percentiles])

                # Process the calculated agg values to response
                for key, value in agg_calculation.items():
                    if not isinstance(value, list):
                        results[key].append(value)
                        continue
                    elif isinstance(value, list) and pd_aggs == ["quantile"]:
                        results[f"{key}_{pd_aggs[0]}"].extend(value)
                    else:
                        for pd_agg, val in zip(pd_aggs, value):
                            results[f"{key}_{pd_agg}"].append(val)

        if pd_aggs == ["quantile"] and len_percentiles > 1:
            # by never holds None by default, we make an exception
            # here to maintain output same as pandas, also mypy complains
            by = by + [None]  # type: ignore

        agg_df = pd.DataFrame(results).set_index(by).sort_index()

        if is_dataframe_agg:
            # Convert header columns to MultiIndex
            agg_df.columns = pd.MultiIndex.from_product([headers, pd_aggs])
        else:
            # Convert header columns to Index
            agg_df.columns = pd.Index(headers)

        return agg_df

    @staticmethod
    def bucket_generator(
        query_compiler: "QueryCompiler", body: "Query"
    ) -> Generator[Sequence[Dict[str, Any]], None, Sequence[Dict[str, Any]]]:
        """
            This can be used for all groupby operations.
        e.g.
        "aggregations": {
            "groupby_buckets": {
                "after_key": {"total_quantity": 8},
                "buckets": [
                    {
                        "key": {"total_quantity": 1},
                        "doc_count": 87,
                        "taxful_total_price_avg": {"value": 48.035978536496216},
                    }
                ],
            }
        }
        Returns
        -------
        A generator which initially yields the bucket
        If after_key is found, use it to fetch the next set of buckets.

        """
        while True:
            res = query_compiler._client.search(
                index=query_compiler._index_pattern,
                size=0,
                body=body.to_search_body(),
            )

            # Pagination Logic
            composite_buckets: Dict[str, Any] = res["aggregations"]["groupby_buckets"]

            after_key: Optional[Dict[str, Any]] = composite_buckets.get(
                "after_key", None
            )
            buckets: Sequence[Dict[str, Any]] = composite_buckets["buckets"]

            if after_key:

                # yield the bucket which contains the result
                yield buckets

                body.composite_agg_after_key(
                    name="groupby_buckets",
                    after_key=after_key,
                )
            else:
                return buckets

    @staticmethod
    def _map_pd_aggs_to_es_aggs(
        pd_aggs: List[str], percentiles: Optional[List[float]] = None
    ) -> Union[List[str], List[Tuple[str, List[float]]]]:
        """
        Args:
            pd_aggs - list of pandas aggs (e.g. ['mad', 'min', 'std'] etc.)
            percentiles - list of percentiles for 'quantile' agg

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
        # pd aggs that will be mapped to es aggs
        # that can use 'extended_stats'.
        extended_stats_pd_aggs = {"mean", "min", "max", "sum", "var", "std"}
        extended_stats_es_aggs = {"avg", "min", "max", "sum"}
        extended_stats_calls = 0

        es_aggs: List[Any] = []
        for pd_agg in pd_aggs:
            if pd_agg in extended_stats_pd_aggs:
                extended_stats_calls += 1

            # Aggs that are 'extended_stats' compatible
            if pd_agg == "count":
                es_aggs.append("value_count")
            elif pd_agg == "max":
                es_aggs.append("max")
            elif pd_agg == "min":
                es_aggs.append("min")
            elif pd_agg == "mean":
                es_aggs.append("avg")
            elif pd_agg == "sum":
                es_aggs.append("sum")
            elif pd_agg == "std":
                es_aggs.append(("extended_stats", "std_deviation"))
            elif pd_agg == "var":
                es_aggs.append(("extended_stats", "variance"))

            # Aggs that aren't 'extended_stats' compatible
            elif pd_agg == "nunique":
                es_aggs.append("cardinality")
            elif pd_agg == "mad":
                es_aggs.append("median_absolute_deviation")
            elif pd_agg == "median":
                es_aggs.append(("percentiles", (50.0,)))
            elif pd_agg == "quantile":
                # None when 'quantile' is called in df.agg[...]
                # Behaves same as median because pandas does the same.
                if percentiles is not None:
                    es_aggs.append(("percentiles", tuple(percentiles)))
                else:
                    es_aggs.append(("percentiles", (50.0,)))

            elif pd_agg == "mode":
                if len(pd_aggs) != 1:
                    raise NotImplementedError(
                        "Currently mode is not supported in df.agg(...). Try df.mode()"
                    )
                else:
                    es_aggs.append("mode")

            # Not implemented
            elif pd_agg == "rank":
                # TODO
                raise NotImplementedError(pd_agg, " not currently implemented")
            elif pd_agg == "sem":
                # TODO
                raise NotImplementedError(pd_agg, " not currently implemented")
            else:
                raise NotImplementedError(pd_agg, " not currently implemented")

        # If two aggs compatible with 'extended_stats' is called we can
        # piggy-back on that single aggregation.
        if extended_stats_calls >= 2:
            es_aggs = [
                ("extended_stats", es_agg)
                if es_agg in extended_stats_es_aggs
                else es_agg
                for es_agg in es_aggs
            ]

        return es_aggs

    def filter(
        self,
        query_compiler: "QueryCompiler",
        items: Optional[List[str]] = None,
        like: Optional[str] = None,
        regex: Optional[str] = None,
    ) -> None:
        # This function is only called for axis='index',
        # DataFrame.filter(..., axis="columns") calls .drop()
        if items is not None:
            self.filter_index_values(
                query_compiler, field=query_compiler.index.es_index_field, items=items
            )
            return
        elif like is not None:
            arg_name = "like"
        else:
            assert regex is not None
            arg_name = "regex"

        raise NotImplementedError(
            f".filter({arg_name}='...', axis='index') is currently not supported due "
            f"to substring and regex operations not being available for Elasticsearch document IDs."
        )

    def describe(self, query_compiler: "QueryCompiler") -> pd.DataFrame:
        query_params, post_processing = self._resolve_tasks(query_compiler)

        size = self._size(query_params, post_processing)
        if size is not None:
            raise NotImplementedError(
                f"Can not count field matches if size is set {size}"
            )

        df1 = self.aggs(
            query_compiler=query_compiler,
            pd_aggs=["count", "mean", "std", "min", "max"],
            numeric_only=True,
        )
        df2 = self.quantile(
            query_compiler=query_compiler,
            pd_aggs=["quantile"],
            quantiles=[0.25, 0.5, 0.75],
            is_dataframe=True,
            numeric_only=True,
        )

        # Convert [.25,.5,.75] to ["25%", "50%", "75%"]
        df2 = df2.set_index([["25%", "50%", "75%"]])

        return pd.concat([df1, df2]).reindex(
            ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
        )

    def to_pandas(
        self, query_compiler: "QueryCompiler", show_progress: bool = False
    ) -> None:

        collector = PandasDataFrameCollector(show_progress)

        self._es_results(query_compiler, collector)

        return collector._df

    def to_csv(
        self,
        query_compiler: "QueryCompiler",
        show_progress: bool = False,
        **kwargs: Union[bool, str],
    ) -> None:

        collector = PandasToCSVCollector(show_progress, **kwargs)

        self._es_results(query_compiler, collector)

        return collector._ret

    def _es_results(
        self,
        query_compiler: "QueryCompiler",
        collector: Union["PandasToCSVCollector", "PandasDataFrameCollector"],
    ) -> None:
        query_params, post_processing = self._resolve_tasks(query_compiler)

        result_size, sort_params = Operations._query_params_to_size_and_sort(
            query_params
        )

        script_fields = query_params.script_fields
        query = Query(query_params.query)

        body = query.to_search_body()
        if script_fields is not None:
            body["script_fields"] = script_fields

        # Only return requested field_names and add them to body
        _source = query_compiler.get_field_names(include_scripted_fields=False)
        body["_source"] = _source if _source else False

        if sort_params:
            body["sort"] = [sort_params]

        es_results = list(
            search_yield_hits(
                query_compiler=query_compiler, body=body, max_number_of_hits=result_size
            )
        )

        _, df = query_compiler._es_results_to_pandas(es_results)
        df = self._apply_df_post_processing(df, post_processing)
        collector.collect(df)

    def index_count(self, query_compiler: "QueryCompiler", field: str) -> int:
        # field is the index field so count values
        query_params, post_processing = self._resolve_tasks(query_compiler)

        size = self._size(query_params, post_processing)

        # Size is dictated by operations
        if size is not None:
            # TODO - this is not necessarily valid as the field may not exist in ALL these docs
            return size

        body = Query(query_params.query)
        body.exists(field, must=True)

        count: int = query_compiler._client.count(
            index=query_compiler._index_pattern, body=body.to_count_body()
        )["count"]
        return count

    def _validate_index_operation(
        self, query_compiler: "QueryCompiler", items: List[str]
    ) -> RESOLVED_TASK_TYPE:
        if not isinstance(items, list):
            raise TypeError(f"list item required - not {type(items)}")

        # field is the index field so count values
        query_params, post_processing = self._resolve_tasks(query_compiler)

        size = self._size(query_params, post_processing)

        # Size is dictated by operations
        if size is not None:
            raise NotImplementedError(
                f"Can not count field matches if size is set {size}"
            )

        return query_params, post_processing

    def index_matches_count(
        self, query_compiler: "QueryCompiler", field: str, items: List[Any]
    ) -> int:
        query_params, post_processing = self._validate_index_operation(
            query_compiler, items
        )

        body = Query(query_params.query)

        if field == Index.ID_INDEX_FIELD:
            body.ids(items, must=True)
        else:
            body.terms(field, items, must=True)

        count: int = query_compiler._client.count(
            index=query_compiler._index_pattern, body=body.to_count_body()
        )["count"]
        return count

    def drop_index_values(
        self, query_compiler: "QueryCompiler", field: str, items: List[str]
    ) -> None:
        self._validate_index_operation(query_compiler, items)

        # Putting boolean queries together
        # i = 10
        # not i = 20
        # _id in [1,2,3]
        # _id not in [1,2,3]
        # a in ['a','b','c']
        # b not in ['a','b','c']
        # For now use term queries
        task: Union["QueryIdsTask", "QueryTermsTask"]
        if field == Index.ID_INDEX_FIELD:
            task = QueryIdsTask(False, items)
        else:
            task = QueryTermsTask(False, field, items)
        self._tasks.append(task)

    def filter_index_values(
        self, query_compiler: "QueryCompiler", field: str, items: List[str]
    ) -> None:
        # Basically .drop_index_values() except with must=True on tasks.
        self._validate_index_operation(query_compiler, items)

        task: Union["QueryIdsTask", "QueryTermsTask"]
        if field == Index.ID_INDEX_FIELD:
            task = QueryIdsTask(True, items, sort_index_by_ids=True)
        else:
            task = QueryTermsTask(True, field, items)
        self._tasks.append(task)

    @staticmethod
    def _query_params_to_size_and_sort(
        query_params: QueryParams,
    ) -> Tuple[Optional[int], Optional[Dict[str, str]]]:
        sort_params = None
        if query_params.sort_field and query_params.sort_order:
            sort_params = {
                query_params.sort_field: SortOrder.to_string(query_params.sort_order)
            }
        size = query_params.size
        return size, sort_params

    @staticmethod
    def _count_post_processing(
        post_processing: List["PostProcessingAction"],
    ) -> Optional[int]:
        size = None
        for action in post_processing:
            if isinstance(action, SizeTask):
                if size is None or action.size() < size:
                    size = action.size()
        return size

    @staticmethod
    def _apply_df_post_processing(
        df: "pd.DataFrame", post_processing: List["PostProcessingAction"]
    ) -> pd.DataFrame:
        for action in post_processing:
            df = action.resolve_action(df)

        return df

    def _resolve_tasks(self, query_compiler: "QueryCompiler") -> RESOLVED_TASK_TYPE:
        # We now try and combine all tasks into an Elasticsearch query
        # Some operations can be simply combined into a single query
        # other operations require pre-queries and then combinations
        # other operations require in-core post-processing of results
        query_params = QueryParams()
        post_processing: List["PostProcessingAction"] = []

        for task in self._tasks:
            query_params, post_processing = task.resolve_task(
                query_params, post_processing, query_compiler
            )

        if self._arithmetic_op_fields_task is not None:
            (
                query_params,
                post_processing,
            ) = self._arithmetic_op_fields_task.resolve_task(
                query_params, post_processing, query_compiler
            )

        return query_params, post_processing

    def _size(
        self, query_params: "QueryParams", post_processing: List["PostProcessingAction"]
    ) -> Optional[int]:
        # Shrink wrap code around checking if size parameter is set
        size = query_params.size

        pp_size = self._count_post_processing(post_processing)
        if pp_size is not None:
            if size is not None:
                size = min(size, pp_size)
            else:
                size = pp_size

        # This can return None
        return size

    def es_info(self, query_compiler: "QueryCompiler", buf: TextIO) -> None:
        buf.write("Operations:\n")
        buf.write(f" tasks: {self._tasks}\n")

        query_params, post_processing = self._resolve_tasks(query_compiler)
        size, sort_params = Operations._query_params_to_size_and_sort(query_params)
        _source = query_compiler._mappings.get_field_names()

        script_fields = query_params.script_fields
        query = Query(query_params.query)
        body = query.to_search_body()
        if script_fields is not None:
            body["script_fields"] = script_fields

        buf.write(f" size: {size}\n")
        buf.write(f" sort_params: {sort_params}\n")
        buf.write(f" _source: {_source}\n")
        buf.write(f" body: {body}\n")
        buf.write(f" post_processing: {post_processing}\n")

    def update_query(self, boolean_filter: "BooleanFilter") -> None:
        task = BooleanFilterTask(boolean_filter)
        self._tasks.append(task)


def quantile_to_percentile(quantile: Union[int, float]) -> float:
    # To verify if quantile range falls between 0 to 1
    if isinstance(quantile, (int, float)):
        quantile = float(quantile)
        if quantile > 1 or quantile < 0:
            raise ValueError(
                f"quantile should be in range of 0 and 1, given {quantile}"
            )
    else:
        raise TypeError("quantile should be of type int or float")
    # quantile * 100 = percentile
    # return float(...) because min(1.0) gives 1
    return float(min(100, max(0, quantile * 100)))


class PandasToCSVCollector:
    def __init__(self, show_progress: bool, **kwargs: Union[bool, str]) -> None:
        self._args = kwargs
        self._show_progress = show_progress
        self._ret = None
        self._first_time = True

    def collect(self, df: "pd.DataFrame") -> None:
        # If this is the first time we collect results, then write header, otherwise don't write header
        # and append results
        if self._first_time:
            self._first_time = False
            df.to_csv(**self._args)
        else:
            # Don't write header, and change mode to append
            self._args["header"] = False
            self._args["mode"] = "a"
            df.to_csv(**self._args)

    @staticmethod
    def batch_size() -> int:
        # By default read n docs and then dump to csv
        batch_size: int = DEFAULT_CSV_BATCH_OUTPUT_SIZE
        return batch_size

    @property
    def show_progress(self) -> bool:
        return self._show_progress


class PandasDataFrameCollector:
    def __init__(self, show_progress: bool) -> None:
        self._df = None
        self._show_progress = show_progress

    def collect(self, df: "pd.DataFrame") -> None:
        # This collector does not batch data on output. Therefore, batch_size is fixed to None and this method
        # is only called once.
        if self._df is not None:
            raise RuntimeError(
                "Logic error in execution, this method must only be called once for this"
                "collector - batch_size == None"
            )
        self._df = df

    @staticmethod
    def batch_size() -> None:
        # Do not change (see notes on collect)
        return None

    @property
    def show_progress(self) -> bool:
        return self._show_progress


def search_yield_hits(
    query_compiler: "QueryCompiler",
    body: Dict[str, Any],
    max_number_of_hits: Optional[int],
) -> Generator[Dict[str, Any], None, None]:
    """
    This is a generator used to initialize point in time API and query the
    search API and return generator which yields an individual documents

    Parameters
    ----------
    query_compiler:
        An instance of query_compiler
    body:
        body for search API
    max_number_of_hits: Optional[int]
        Maximum number of documents to yield, set to 'None' to
        yield all documents.

    Examples
    --------
    >>> results = list(search_yield_hits(query_compiler, body, 2)) # doctest: +SKIP
    [{'_index': 'flights', '_type': '_doc', '_id': '0', '_score': None, '_source': {...}, 'sort': [...]},
    {'_index': 'flights', '_type': '_doc', '_id': '1', '_score': None, '_source': {...}, 'sort': [...]}]
    """
    # Make a copy of 'body' to avoid mutating it outside this function.
    body = body.copy()

    # Use the default search size
    body.setdefault("size", DEFAULT_SEARCH_SIZE)

    # Elasticsearch 7.12 added '_shard_doc' sort tiebreaker for PITs which
    # means we're guaranteed to be safe on documents with a duplicate sort rank.
    if es_version(query_compiler._client) >= (7, 12, 0):
        yield from _search_with_pit_and_search_after(
            query_compiler=query_compiler,
            body=body,
            max_number_of_hits=max_number_of_hits,
        )

    # Otherwise we use 'scroll' like we used to.
    else:
        yield from _search_with_scroll(
            query_compiler=query_compiler,
            body=body,
            max_number_of_hits=max_number_of_hits,
        )


def _search_with_scroll(
    query_compiler: "QueryCompiler",
    body: Dict[str, Any],
    max_number_of_hits: Optional[int],
) -> Generator[Dict[str, Any], None, None]:
    # No documents, no reason to send a search.
    if max_number_of_hits == 0:
        return

    client = query_compiler._client
    hits_yielded = 0

    # Make the initial search with 'scroll' set
    resp = client.search(
        index=query_compiler._index_pattern,
        body=body,
        scroll=DEFAULT_PIT_KEEP_ALIVE,
    )
    scroll_id: Optional[str] = resp.get("_scroll_id", None)

    try:
        while scroll_id and (
            max_number_of_hits is None or hits_yielded < max_number_of_hits
        ):
            hits: List[Dict[str, Any]] = resp["hits"]["hits"]

            # If we didn't receive any hits it means we've reached the end.
            if not hits:
                break

            # Calculate which hits should be yielded from this batch
            if max_number_of_hits is None:
                hits_to_yield = len(hits)
            else:
                hits_to_yield = min(len(hits), max_number_of_hits - hits_yielded)

            # Yield the hits we need to and then track the total number.
            yield from hits[:hits_to_yield]
            hits_yielded += hits_to_yield

            # Retrieve the next set of results
            resp = client.scroll(
                body={"scroll_id": scroll_id, "scroll": DEFAULT_PIT_KEEP_ALIVE},
            )
            scroll_id = resp.get("_scroll_id", None)  # Update the scroll ID.

    finally:
        # Close the scroll if we have one open
        if scroll_id is not None:
            try:
                client.clear_scroll(body={"scroll_id": [scroll_id]})
            except NotFoundError:
                pass


def _search_with_pit_and_search_after(
    query_compiler: "QueryCompiler",
    body: Dict[str, Any],
    max_number_of_hits: Optional[int],
) -> Generator[Dict[str, Any], None, None]:

    # No documents, no reason to send a search.
    if max_number_of_hits == 0:
        return

    client = query_compiler._client
    hits_yielded = 0  # Track the total number of hits yielded.
    pit_id: Optional[str] = None

    # Pagination with 'search_after' must have a 'sort' setting.
    # Using '_doc:asc' is the most efficient as reads documents
    # in the order that they're written on disk in Lucene.
    body.setdefault("sort", [{"_doc": "asc"}])

    # Improves performance by not tracking # of hits. We only
    # care about the hit itself for these queries.
    body.setdefault("track_total_hits", False)

    try:
        pit_id = client.open_point_in_time(
            index=query_compiler._index_pattern, keep_alive=DEFAULT_PIT_KEEP_ALIVE
        )["id"]

        # Modify the search with the new point in time ID and keep-alive time.
        body["pit"] = {"id": pit_id, "keep_alive": DEFAULT_PIT_KEEP_ALIVE}

        while max_number_of_hits is None or hits_yielded < max_number_of_hits:
            resp = client.search(body=body)
            hits: List[Dict[str, Any]] = resp["hits"]["hits"]

            # The point in time ID can change between searches so we
            # need to keep the next search up-to-date
            pit_id = resp.get("pit_id", pit_id)
            body["pit"]["id"] = pit_id

            # If we didn't receive any hits it means we've reached the end.
            if not hits:
                break

            # Calculate which hits should be yielded from this batch
            if max_number_of_hits is None:
                hits_to_yield = len(hits)
            else:
                hits_to_yield = min(len(hits), max_number_of_hits - hits_yielded)

            # Yield the hits we need to and then track the total number.
            yield from hits[:hits_to_yield]
            hits_yielded += hits_to_yield

            # Set the 'search_after' for the next request
            # to be the last sort value for this set of hits.
            body["search_after"] = hits[-1]["sort"]

    finally:
        # We want to cleanup the point in time if we allocated one
        # to keep our memory footprint low.
        if pit_id is not None:
            try:
                client.close_point_in_time(body={"id": pit_id})
            except NotFoundError:
                # If a point in time is already closed Elasticsearch throws NotFoundError
                pass
