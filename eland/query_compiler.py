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
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    TextIO,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd  # type: ignore

from eland.common import (
    DEFAULT_PROGRESS_REPORTING_NUM_ROWS,
    elasticsearch_date_to_pandas_date,
    ensure_es_client,
)
from eland.field_mappings import FieldMappings
from eland.filter import BooleanFilter, QueryFilter
from eland.index import Index
from eland.operations import Operations

if TYPE_CHECKING:
    from elasticsearch import Elasticsearch

    from eland.arithmetics import ArithmeticSeries

    from .tasks import ArithmeticOpFieldsTask  # noqa: F401


class QueryCompiler:
    """
    Some notes on what can and can not be mapped:

    1. df.head(10)

    /_search?size=10

    2. df.tail(10)

    /_search?size=10&sort=_doc:desc
    + post_process results (sort_index)

    3. df[['OriginAirportID', 'AvgTicketPrice', 'Carrier']]

    /_search
    { '_source': ['OriginAirportID', 'AvgTicketPrice', 'Carrier']}

    4. df.drop(['1', '2'])

    /_search
    {'query': {'bool': {'must': [], 'must_not': [{'ids': {'values': ['1', '2']}}]}}, 'aggs': {}}

    This doesn't work is size is set (e.g. head/tail) as we don't know in Elasticsearch if values '1' or '2' are
    in the first/last n fields.

    A way to mitigate this would be to post process this drop - TODO
    """

    def __init__(
        self,
        client: Optional[
            Union[str, List[str], Tuple[str, ...], "Elasticsearch"]
        ] = None,
        index_pattern: Optional[str] = None,
        display_names=None,
        index_field=None,
        to_copy=None,
    ) -> None:
        # Implement copy as we don't deep copy the client
        if to_copy is not None:
            self._client = to_copy._client
            self._index_pattern = to_copy._index_pattern
            self._index: "Index" = Index(self, to_copy._index.es_index_field)
            self._operations: "Operations" = copy.deepcopy(to_copy._operations)
            self._mappings: FieldMappings = copy.deepcopy(to_copy._mappings)
        else:
            self._client = ensure_es_client(client)
            self._index_pattern = index_pattern
            # Get and persist mappings, this allows us to correctly
            # map returned types from Elasticsearch to pandas datatypes
            self._mappings = FieldMappings(
                client=self._client,
                index_pattern=self._index_pattern,
                display_names=display_names,
            )
            self._index = Index(self, index_field)
            self._operations = Operations()

    @property
    def index(self) -> Index:
        return self._index

    @property
    def columns(self) -> pd.Index:
        columns = self._mappings.display_names

        return pd.Index(columns)

    def _get_display_names(self) -> "pd.Index":
        display_names = self._mappings.display_names

        return pd.Index(display_names)

    def _set_display_names(self, display_names: List[str]) -> None:
        self._mappings.display_names = display_names

    def get_field_names(self, include_scripted_fields: bool) -> List[str]:
        return self._mappings.get_field_names(include_scripted_fields)

    def add_scripted_field(self, scripted_field_name, display_name, pd_dtype):
        result = self.copy()
        self._mappings.add_scripted_field(scripted_field_name, display_name, pd_dtype)
        return result

    @property
    def dtypes(self) -> pd.Series:
        return self._mappings.dtypes()

    @property
    def es_dtypes(self) -> pd.Series:
        return self._mappings.es_dtypes()

    # END Index, columns, and dtypes objects

    def _es_results_to_pandas(
        self,
        results: List[Dict[str, Any]],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
    ) -> "pd.Dataframe":
        """
        Parameters
        ----------
        results: List[Dict[str, Any]]
            Elasticsearch results from self.client.search

        Returns
        -------
        df: pandas.DataFrame
            _source values extracted from results and mapped to pandas DataFrame
            dtypes are mapped via Mapping object

        Notes
        -----
        Fields containing lists in Elasticsearch don't map easily to pandas.DataFrame
        For example, an index with mapping:
        ```
        "mappings" : {
          "properties" : {
            "group" : {
              "type" : "keyword"
            },
            "user" : {
              "type" : "nested",
              "properties" : {
                "first" : {
                  "type" : "keyword"
                },
                "last" : {
                  "type" : "keyword"
                }
              }
            }
          }
        }
        ```
        Adding a document:
        ```
        "_source" : {
          "group" : "amsterdam",
          "user" : [
            {
              "first" : "John",
              "last" : "Smith"
            },
            {
              "first" : "Alice",
              "last" : "White"
            }
          ]
        }
        ```
        (https://www.elastic.co/guide/en/elasticsearch/reference/current/nested.html)
        this would be transformed internally (in Elasticsearch) into a document that looks more like this:
        ```
        {
          "group" :        "amsterdam",
          "user.first" : [ "alice", "john" ],
          "user.last" :  [ "smith", "white" ]
        }
        ```
        When mapping this a pandas data frame we mimic this transformation.

        Similarly, if a list is added to Elasticsearch:
        ```
        PUT my_index/_doc/1
        {
          "list" : [
            0, 1, 2
          ]
        }
        ```
        The mapping is:
        ```
        "mappings" : {
          "properties" : {
            "user" : {
              "type" : "long"
            }
          }
        }
        ```
        TODO - explain how lists are handled
            (https://www.elastic.co/guide/en/elasticsearch/reference/current/array.html)
        TODO - an option here is to use Elasticsearch's multi-field matching instead of pandas treatment of lists
            (which isn't great)
        NOTE - using this lists is generally not a good way to use this API
        """
        partial_result = False

        if results is None:
            return partial_result, self._empty_pd_ef()

        # This is one of the most performance critical areas of eland, and it repeatedly calls
        # self._mappings.field_name_pd_dtype and self._mappings.date_field_format
        # therefore create a simple cache for this data
        field_mapping_cache = FieldMappingCache(self._mappings)

        rows = []
        index = []

        i = 0
        for hit in results:
            i = i + 1

            if "_source" in hit:
                row = hit["_source"]
            else:
                row = {}

            # script_fields appear in 'fields'
            if "fields" in hit:
                fields = hit["fields"]
                for key, value in fields.items():
                    row[key] = value

            # get index value - can be _id or can be field value in source
            if self._index.is_source_field:
                index_field = row[self._index.es_index_field]
            else:
                index_field = hit[self._index.es_index_field]
            index.append(index_field)

            # flatten row to map correctly to 2D DataFrame
            rows.append(self._flatten_dict(row, field_mapping_cache))

            if batch_size is not None:
                if i >= batch_size:
                    partial_result = True
                    break

            if show_progress:
                if i % DEFAULT_PROGRESS_REPORTING_NUM_ROWS == 0:
                    print(f"{datetime.now()}: read {i} rows")

        # Create pandas DataFrame
        df = pd.DataFrame(data=rows, index=index)

        # _source may not contain all field_names in the mapping
        # therefore, fill in missing field_names
        # (note this returns self.field_names NOT IN df.columns)
        missing_field_names = list(
            set(self.get_field_names(include_scripted_fields=True)) - set(df.columns)
        )

        for missing in missing_field_names:
            pd_dtype = self._mappings.field_name_pd_dtype(missing)
            df[missing] = pd.Series(dtype=pd_dtype)

        # Rename columns
        df.rename(columns=self._mappings.get_renames(), inplace=True)

        # Sort columns in mapping order
        if len(self.columns) > 1:
            df = df[self.columns]

        if show_progress:
            print(f"{datetime.now()}: read {i} rows")

        return partial_result, df

    def _flatten_dict(self, y, field_mapping_cache: "FieldMappingCache"):
        out = {}

        def flatten(x, name=""):
            # We flatten into source fields e.g. if type=geo_point
            # location: {lat=52.38, lon=4.90}
            if name == "":
                is_source_field = False
                pd_dtype = "object"
            else:
                try:
                    pd_dtype = field_mapping_cache.field_name_pd_dtype(name[:-1])
                    is_source_field = True
                except KeyError:
                    is_source_field = False
                    pd_dtype = "object"

            if not is_source_field and isinstance(x, dict):
                for a in x:
                    flatten(x[a], name + a + ".")
            elif not is_source_field and isinstance(x, list):
                for a in x:
                    flatten(a, name)
            elif is_source_field:  # only print source fields from mappings
                # (TODO - not so efficient for large number of fields and filtered mapping)
                field_name = name[:-1]

                # Coerce types - for now just datetime
                if pd_dtype == "datetime64[ns]":
                    x = elasticsearch_date_to_pandas_date(
                        x, field_mapping_cache.date_field_format(field_name)
                    )

                # Elasticsearch can have multiple values for a field. These are represented as lists, so
                # create lists for this pivot (see notes above)
                if field_name in out:
                    if not isinstance(out[field_name], list):
                        field_as_list = [out[field_name]]
                        out[field_name] = field_as_list
                    out[field_name].append(x)
                else:
                    out[field_name] = x
            else:
                # Script fields end up here

                # Elasticsearch returns 'Infinity' as a string for np.inf values.
                # Map this to a numeric value to avoid this whole Series being classed as an object
                # TODO - create a lookup for script fields and dtypes to only map 'Infinity'
                #        if the field is numeric. This implementation will currently map
                #        any script field with "Infinity" as a string to np.inf
                if x == "Infinity":
                    out[name[:-1]] = np.inf
                else:
                    out[name[:-1]] = x

        flatten(y)

        return out

    def _index_count(self) -> int:
        """
        Returns
        -------
        index_count: int
            Count of docs where index_field exists
        """
        return self._operations.index_count(self, self.index.es_index_field)

    def _index_matches_count(self, items: List[Any]) -> int:
        """
        Returns
        -------
        index_count: int
            Count of docs where items exist
        """
        return self._operations.index_matches_count(
            self, self.index.es_index_field, items
        )

    def _empty_pd_ef(self):
        # Return an empty dataframe with correct columns and dtypes
        df = pd.DataFrame()
        for c, d in zip(self.dtypes.index, self.dtypes.values):
            df[c] = pd.Series(dtype=d)
        return df

    def copy(self) -> "QueryCompiler":
        return QueryCompiler(to_copy=self)

    def rename(self, renames, inplace: bool = False) -> "QueryCompiler":
        if inplace:
            self._mappings.rename(renames)
            return self
        else:
            result = self.copy()
            result._mappings.rename(renames)
            return result

    def head(self, n: int) -> "QueryCompiler":
        result = self.copy()

        result._operations.head(self._index, n)

        return result

    def tail(self, n: int) -> "QueryCompiler":
        result = self.copy()

        result._operations.tail(self._index, n)

        return result

    def sample(
        self, n: Optional[int] = None, frac=None, random_state=None
    ) -> "QueryCompiler":
        result = self.copy()

        if n is None and frac is None:
            n = 1
        elif n is None and frac is not None:
            index_length = self._index_count()
            n = int(round(frac * index_length))

        if n < 0:
            raise ValueError(
                "A negative number of rows requested. Please provide positive value."
            )

        result._operations.sample(self._index, n, random_state)

        return result

    def es_match(
        self,
        text: str,
        columns: Sequence[str],
        *,
        match_phrase: bool = False,
        match_only_text_fields: bool = True,
        multi_match_type: Optional[str] = None,
        analyzer: Optional[str] = None,
        fuzziness: Optional[Union[int, str]] = None,
        **kwargs: Any,
    ) -> QueryFilter:
        if len(columns) < 1:
            raise ValueError("columns can't be empty")

        es_dtypes = self.es_dtypes.to_dict()

        # Build the base options for the 'match_*' query
        options = {"query": text}
        if analyzer is not None:
            options["analyzer"] = analyzer
        if fuzziness is not None:
            options["fuzziness"] = fuzziness
        options.update(kwargs)

        # Warn the user if they're not querying text columns
        if match_only_text_fields:
            non_text_columns = {}
            for column in columns:
                # Don't worry about wildcards
                if "*" in column:
                    continue

                es_dtype = es_dtypes[column]
                if es_dtype != "text":
                    non_text_columns[column] = es_dtype
            if non_text_columns:
                raise ValueError(
                    f"Attempting to run es_match() on non-text fields "
                    f"({', '.join([k + '=' + v for k, v in non_text_columns.items()])}) "
                    f"means that these fields may not be analyzed properly. "
                    f"Consider reindexing these fields as text or use 'match_only_text_es_dtypes=False' "
                    f"to use match anyways"
                )
        else:
            options.setdefault("lenient", True)

        # If only one column use 'match'
        # otherwise use 'multi_match' with 'fields'
        if len(columns) == 1:
            if multi_match_type is not None:
                raise ValueError(
                    "multi_match_type parameter only valid "
                    "when searching more than one column"
                )
            query = {"match_phrase" if match_phrase else "match": {columns[0]: options}}
        else:
            options["fields"] = columns
            if match_phrase:
                if multi_match_type not in ("phrase", None):
                    raise ValueError(
                        f"match_phrase=True and multi_match_type={multi_match_type!r} "
                        f"are not compatible. Must be multi_match_type='phrase'"
                    )
                multi_match_type = "phrase"
            if multi_match_type is not None:
                options["type"] = multi_match_type

            query = {"multi_match": options}
        return QueryFilter(query)

    def es_query(self, query: Dict[str, Any]) -> "QueryCompiler":
        return self._update_query(QueryFilter(query))

    # To/From Pandas
    def to_pandas(self, show_progress: bool = False):
        """Converts Eland DataFrame to Pandas DataFrame.

        Returns:
            Pandas DataFrame
        """
        return self._operations.to_pandas(self, show_progress)

    # To CSV
    def to_csv(self, **kwargs):
        """Serialises Eland Dataframe to CSV

        Returns:
            If path_or_buf is None, returns the resulting csv format as a string. Otherwise returns None.
        """
        return self._operations.to_csv(self, **kwargs)

    # __getitem__ methods
    def getitem_column_array(self, key, numeric=False):
        """Get column data for target labels.

        Args:
            key: Target labels by which to retrieve data.
            numeric: A boolean representing whether or not the key passed in represents
                the numeric index or the named index.

        Returns:
            A new QueryCompiler.
        """
        result = self.copy()

        if numeric:
            raise NotImplementedError("Not implemented yet...")

        result._mappings.display_names = list(key)

        return result

    def drop(
        self, index: Optional[str] = None, columns: Optional[List[str]] = None
    ) -> "QueryCompiler":
        result = self.copy()

        # Drop gets all columns and removes drops
        if columns is not None:
            # columns is a pandas.Index so we can use pandas drop feature
            new_columns = self.columns.drop(columns)
            result._mappings.display_names = new_columns.to_list()

        if index is not None:
            result._operations.drop_index_values(self, self.index.es_index_field, index)

        return result

    def filter(
        self,
        items: Optional[List[str]] = None,
        like: Optional[str] = None,
        regex: Optional[str] = None,
    ) -> "QueryCompiler":
        # field will be es_index_field for DataFrames or the column for Series.
        # This function is only called for axis='index',
        # DataFrame.filter(..., axis="columns") calls .drop()
        result = self.copy()
        result._operations.filter(self, items=items, like=like, regex=regex)
        return result

    def aggs(
        self, func: List[str], numeric_only: Optional[bool] = None
    ) -> pd.DataFrame:
        return self._operations.aggs(self, func, numeric_only=numeric_only)

    def count(self) -> pd.Series:
        return self._operations.count(self)

    def mean(self, numeric_only: Optional[bool] = None) -> pd.Series:
        return self._operations._metric_agg_series(
            self, ["mean"], numeric_only=numeric_only
        )

    def var(self, numeric_only: Optional[bool] = None) -> pd.Series:
        return self._operations._metric_agg_series(
            self, ["var"], numeric_only=numeric_only
        )

    def std(self, numeric_only: Optional[bool] = None) -> pd.Series:
        return self._operations._metric_agg_series(
            self, ["std"], numeric_only=numeric_only
        )

    def mad(self, numeric_only: Optional[bool] = None) -> pd.Series:
        return self._operations._metric_agg_series(
            self, ["mad"], numeric_only=numeric_only
        )

    def median(self, numeric_only: Optional[bool] = None) -> pd.Series:
        return self._operations._metric_agg_series(
            self, ["median"], numeric_only=numeric_only
        )

    def sum(self, numeric_only: Optional[bool] = None) -> pd.Series:
        return self._operations._metric_agg_series(
            self, ["sum"], numeric_only=numeric_only
        )

    def min(self, numeric_only: Optional[bool] = None) -> pd.Series:
        return self._operations._metric_agg_series(
            self, ["min"], numeric_only=numeric_only
        )

    def max(self, numeric_only: Optional[bool] = None) -> pd.Series:
        return self._operations._metric_agg_series(
            self, ["max"], numeric_only=numeric_only
        )

    def nunique(self) -> pd.Series:
        return self._operations._metric_agg_series(
            self, ["nunique"], numeric_only=False
        )

    def mode(
        self,
        es_size: int,
        numeric_only: bool = False,
        dropna: bool = True,
        is_dataframe: bool = True,
    ) -> Union[pd.DataFrame, pd.Series]:
        return self._operations.mode(
            self,
            pd_aggs=["mode"],
            numeric_only=numeric_only,
            dropna=dropna,
            is_dataframe=is_dataframe,
            es_size=es_size,
        )

    def quantile(
        self,
        quantiles: Union[int, float, List[int], List[float]],
        numeric_only: Optional[bool] = True,
        is_dataframe: bool = True,
    ) -> Union[pd.DataFrame, pd.Series, Any]:
        """
        Holds quantile object for both DataFrame and Series

        Parameters
        ----------
        quantiles:
            list of quantiles for computation
        numeric_only:
            Flag used to filter numeric columns
        is_dataframe:
            To identify if quantile is called from Series or DataFrame
            True: Called from DataFrame
            False: Called from Series

        """
        return self._operations.quantile(
            self,
            pd_aggs=["quantile"],
            quantiles=quantiles,
            numeric_only=numeric_only,
            is_dataframe=is_dataframe,
        )

    def aggs_groupby(
        self,
        by: List[str],
        pd_aggs: List[str],
        dropna: bool = True,
        is_dataframe_agg: bool = False,
        numeric_only: Optional[bool] = True,
        quantiles: Optional[Union[int, float, List[int], List[float]]] = None,
    ) -> pd.DataFrame:
        return self._operations.aggs_groupby(
            self,
            by=by,
            pd_aggs=pd_aggs,
            quantiles=quantiles,
            dropna=dropna,
            is_dataframe_agg=is_dataframe_agg,
            numeric_only=numeric_only,
        )

    def idx(self, axis: int, sort_order: str) -> pd.Series:
        return self._operations.idx(self, axis=axis, sort_order=sort_order)

    def value_counts(self, es_size: int) -> pd.Series:
        return self._operations.value_counts(self, es_size)

    def es_info(self, buf: TextIO) -> None:
        buf.write(f"es_index_pattern: {self._index_pattern}\n")

        self._index.es_info(buf)
        self._mappings.es_info(buf)
        self._operations.es_info(self, buf)

    def describe(self) -> pd.DataFrame:
        return self._operations.describe(self)

    def _hist(self, num_bins: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self._operations.hist(self, num_bins)

    def _update_query(self, boolean_filter: "BooleanFilter") -> "QueryCompiler":
        result = self.copy()

        result._operations.update_query(boolean_filter)

        return result

    def check_arithmetics(self, right: "QueryCompiler") -> None:
        """
        Compare 2 query_compilers to see if arithmetic operations can be performed by the NDFrame object.

        This does very basic comparisons and ignores some of the complexities of incompatible task lists

        Raises exception if incompatible

        Parameters
        ----------
        right: QueryCompiler
            The query compiler to compare self to

        Raises
        ------
        TypeError, ValueError
            If arithmetic operations aren't possible
        """
        if not isinstance(right, QueryCompiler):
            raise TypeError(f"Incompatible types {type(self)} != {type(right)}")

        if self._client != right._client:
            raise ValueError(
                f"Can not perform arithmetic operations across different clients"
                f"{self._client} != {right._client}"
            )

        if self._index.es_index_field != right._index.es_index_field:
            raise ValueError(
                f"Can not perform arithmetic operations across different index fields "
                f"{self._index.es_index_field} != {right._index.es_index_field}"
            )

        if self._index_pattern != right._index_pattern:
            raise ValueError(
                f"Can not perform arithmetic operations across different index patterns"
                f"{self._index_pattern} != {right._index_pattern}"
            )

    def arithmetic_op_fields(
        self, display_name: str, arithmetic_object: "ArithmeticSeries"
    ) -> "QueryCompiler":
        result = self.copy()

        # create a new field name for this display name
        scripted_field_name = f"script_field_{display_name}"

        # add scripted field
        result._mappings.add_scripted_field(
            scripted_field_name, display_name, arithmetic_object.dtype.name  # type: ignore
        )

        result._operations.arithmetic_op_fields(scripted_field_name, arithmetic_object)

        return result

    def get_arithmetic_op_fields(self) -> Optional["ArithmeticOpFieldsTask"]:
        return self._operations.get_arithmetic_op_fields()

    def display_name_to_aggregatable_name(self, display_name: str) -> str:
        aggregatable_field_name = self._mappings.aggregatable_field_name(display_name)
        if aggregatable_field_name is None:
            raise ValueError(
                f"Can not perform arithmetic operations on non aggregatable fields"
                f"{display_name} is not aggregatable."
            )
        return aggregatable_field_name


class FieldMappingCache:
    """
    Very simple dict cache for field mappings. This improves performance > 3 times on large datasets as
    DataFrame access is slower than dict access.
    """

    def __init__(self, mappings: "FieldMappings") -> None:
        self._mappings = mappings

        self._field_name_pd_dtype: Dict[str, str] = dict()
        self._date_field_format: Dict[str, str] = dict()

    def field_name_pd_dtype(self, es_field_name: str) -> str:
        if es_field_name in self._field_name_pd_dtype:
            return self._field_name_pd_dtype[es_field_name]

        pd_dtype = self._mappings.field_name_pd_dtype(es_field_name)

        # cache this
        self._field_name_pd_dtype[es_field_name] = pd_dtype

        return pd_dtype

    def date_field_format(self, es_field_name: str) -> str:
        if es_field_name in self._date_field_format:
            return self._date_field_format[es_field_name]

        es_date_field_format = self._mappings.date_field_format(es_field_name)

        # cache this
        self._date_field_format[es_field_name] = es_date_field_format

        return es_date_field_format
