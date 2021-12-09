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

import re
import sys
import warnings
from io import StringIO
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd  # type: ignore
from pandas.core.common import apply_if_callable, is_bool_indexer  # type: ignore
from pandas.core.computation.eval import eval  # type: ignore
from pandas.core.dtypes.common import is_list_like  # type: ignore
from pandas.core.indexing import check_bool_indexer  # type: ignore
from pandas.io.common import _expand_user, stringify_path  # type: ignore
from pandas.io.formats import console  # type: ignore
from pandas.io.formats import format as fmt
from pandas.io.formats.printing import pprint_thing  # type: ignore
from pandas.util._validators import validate_bool_kwarg  # type: ignore

import eland.plotting as gfx
from eland.common import DEFAULT_NUM_ROWS_DISPLAYED, docstring_parameter
from eland.filter import BooleanFilter
from eland.groupby import DataFrameGroupBy
from eland.ndframe import NDFrame
from eland.series import Series
from eland.utils import is_valid_attr_name

if TYPE_CHECKING:
    from elasticsearch import Elasticsearch

    from .query_compiler import QueryCompiler


class DataFrame(NDFrame):
    """
    Two-dimensional size-mutable, potentially heterogeneous tabular data structure with labeled axes
    (rows and columns) referencing data stored in Elasticsearch indices.
    Where possible APIs mirror pandas.DataFrame APIs.
    The underlying data is stored in Elasticsearch rather than core memory.

    Parameters
    ----------
    es_client: Elasticsearch client argument(s) (e.g. 'http://localhost:9200')
        - elasticsearch-py parameters or
        - elasticsearch-py instance
    es_index_pattern: str
        Elasticsearch index pattern. This can contain wildcards. (e.g. 'flights')
    columns: list of str, optional
        List of DataFrame columns. A subset of the Elasticsearch index's fields.
    es_index_field: str, optional
        The Elasticsearch index field to use as the DataFrame index. Defaults to _id if None is used.

    See Also
    --------
    :pandas_api_docs:`pandas.DataFrame`

    Examples
    --------
    Constructing DataFrame from an Elasticsearch configuration arguments and an Elasticsearch index

    >>> df = ed.DataFrame('http://localhost:9200', 'flights')
    >>> df.head()
       AvgTicketPrice  Cancelled  ... dayOfWeek           timestamp
    0      841.265642      False  ...         0 2018-01-01 00:00:00
    1      882.982662      False  ...         0 2018-01-01 18:27:00
    2      190.636904      False  ...         0 2018-01-01 17:11:14
    3      181.694216       True  ...         0 2018-01-01 10:33:28
    4      730.041778      False  ...         0 2018-01-01 05:13:00
    <BLANKLINE>
    [5 rows x 27 columns]


    Constructing DataFrame from an Elasticsearch client and an Elasticsearch index

    >>> from elasticsearch import Elasticsearch
    >>> es = Elasticsearch("http://localhost:9200")
    >>> df = ed.DataFrame(es_client=es, es_index_pattern='flights', columns=['AvgTicketPrice', 'Cancelled'])
    >>> df.head()
       AvgTicketPrice  Cancelled
    0      841.265642      False
    1      882.982662      False
    2      190.636904      False
    3      181.694216       True
    4      730.041778      False
    <BLANKLINE>
    [5 rows x 2 columns]

    Constructing DataFrame from an Elasticsearch client and an Elasticsearch index, with 'timestamp' as the  DataFrame
    index field
    (TODO - currently index_field must also be a field if not _id)

    >>> df = ed.DataFrame(
    ...     es_client='http://localhost:9200',
    ...     es_index_pattern='flights',
    ...     columns=['AvgTicketPrice', 'timestamp'],
    ...     es_index_field='timestamp'
    ... )
    >>> df.head()
                         AvgTicketPrice           timestamp
    2018-01-01T00:00:00      841.265642 2018-01-01 00:00:00
    2018-01-01T00:02:06      772.100846 2018-01-01 00:02:06
    2018-01-01T00:06:27      159.990962 2018-01-01 00:06:27
    2018-01-01T00:33:31      800.217104 2018-01-01 00:33:31
    2018-01-01T00:36:51      803.015200 2018-01-01 00:36:51
    <BLANKLINE>
    [5 rows x 2 columns]
    """

    def __init__(
        self,
        es_client: Optional[
            Union[str, List[str], Tuple[str, ...], "Elasticsearch"]
        ] = None,
        es_index_pattern: Optional[str] = None,
        columns: Optional[List[str]] = None,
        es_index_field: Optional[str] = None,
        _query_compiler: Optional["QueryCompiler"] = None,
    ) -> None:
        """
        There are effectively 2 constructors:

        1. client, index_pattern, columns, index_field
        2. query_compiler (eland.QueryCompiler)

        The constructor with 'query_compiler' is for internal use only.
        """
        if _query_compiler is None:
            if es_client is None or es_index_pattern is None:
                raise ValueError(
                    "client and index_pattern must be defined in DataFrame constructor"
                )
        # python 3 syntax
        super().__init__(
            es_client=es_client,
            es_index_pattern=es_index_pattern,
            columns=columns,
            es_index_field=es_index_field,
            _query_compiler=_query_compiler,
        )

    @property
    def columns(self) -> pd.Index:
        """
        The column labels of the DataFrame.

        Returns
        -------
        pandas.Index
            Elasticsearch field names as pandas.Index

        See Also
        --------
        :pandas_api_docs:`pandas.DataFrame.columns`

        Examples
        --------
        >>> df = ed.DataFrame('http://localhost:9200', 'flights')
        >>> assert isinstance(df.columns, pd.Index)
        >>> df.columns
        Index(['AvgTicketPrice', 'Cancelled', 'Carrier', 'Dest', 'DestAirportID', 'DestCityName',
        ...   'DestCountry', 'DestLocation', 'DestRegion', 'DestWeather', 'DistanceKilometers',
        ...   'DistanceMiles', 'FlightDelay', 'FlightDelayMin', 'FlightDelayType', 'FlightNum',
        ...   'FlightTimeHour', 'FlightTimeMin', 'Origin', 'OriginAirportID', 'OriginCityName',
        ...   'OriginCountry', 'OriginLocation', 'OriginRegion', 'OriginWeather', 'dayOfWeek',
        ...   'timestamp'],
        ...   dtype='object')
        """
        return self._query_compiler.columns

    @property
    def empty(self) -> bool:
        """Determines if the DataFrame is empty.

        Returns
        -------
        bool
            If DataFrame is empty, return True, if not return False.

        See Also
        --------
        :pandas_api_docs:`pandas.DataFrame.empty`

        Examples
        --------
        >>> df = ed.DataFrame('http://localhost:9200', 'flights')
        >>> df.empty
        False
        """
        return len(self.columns) == 0 or len(self.index) == 0

    def head(self, n: int = 5) -> "DataFrame":
        """
        Return the first n rows.

        This function returns the first n rows for the object based on position.
        The row order is sorted by index field.
        It is useful for quickly testing if your object has the right type of data in it.

        Parameters
        ----------
        n: int, default 5
            Number of rows to select.

        Returns
        -------
        eland.DataFrame
            eland DataFrame filtered on first n rows sorted by index field

        See Also
        --------
        :pandas_api_docs:`pandas.DataFrame.head`

        Examples
        --------
        >>> df = ed.DataFrame('http://localhost:9200', 'flights', columns=['Origin', 'Dest'])
        >>> df.head(3)
                                    Origin                                          Dest
        0        Frankfurt am Main Airport  Sydney Kingsford Smith International Airport
        1  Cape Town International Airport                     Venice Marco Polo Airport
        2        Venice Marco Polo Airport                     Venice Marco Polo Airport
        <BLANKLINE>
        [3 rows x 2 columns]
        """
        return DataFrame(_query_compiler=self._query_compiler.head(n))

    def tail(self, n: int = 5) -> "DataFrame":
        """
        Return the last n rows.

        This function returns the last n rows for the object based on position.
        The row order is sorted by index field.
        It is useful for quickly testing if your object has the right type of data in it.

        Parameters
        ----------
        n: int, default 5
            Number of rows to select.

        Returns
        -------
        eland.DataFrame:
            eland DataFrame filtered on last n rows sorted by index field

        See Also
        --------
        :pandas_api_docs:`pandas.DataFrame.tail`

        Examples
        --------
        >>> df = ed.DataFrame('http://localhost:9200', 'flights', columns=['Origin', 'Dest'])
        >>> df.tail()
                                                                    Origin  \\
        13054                                   Pisa International Airport...
        13055  Winnipeg / James Armstrong Richardson International Airport...
        13056               Licenciado Benito Juarez International Airport...
        13057                                                Itami Airport...
        13058                               Adelaide International Airport...
        <BLANKLINE>
                                                   Dest...
        13054      Xi'an Xianyang International Airport...
        13055                            Zurich Airport...
        13056                         Ukrainka Air Base...
        13057  Ministro Pistarini International Airport...
        13058   Washington Dulles International Airport...
        <BLANKLINE>
        [5 rows x 2 columns]
        """
        return DataFrame(_query_compiler=self._query_compiler.tail(n))

    def sample(
        self,
        n: Optional[int] = None,
        frac: Optional[float] = None,
        random_state: Optional[int] = None,
    ) -> "DataFrame":
        """
        Return n randomly sample rows or the specify fraction of rows

        Parameters
        ----------
        n : int, optional
            Number of documents from index to return. Cannot be used with `frac`.
            Default = 1 if `frac` = None.
        frac : float, optional
            Fraction of axis items to return. Cannot be used with `n`.
        random_state : int, optional
            Seed for the random number generator.

        Returns
        -------
        eland.DataFrame:
            eland DataFrame filtered containing n rows randomly sampled

        See Also
        --------
        :pandas_api_docs:`pandas.DataFrame.sample`
        """

        if frac is not None and not (0.0 < frac <= 1.0):
            raise ValueError("`frac` must be between 0. and 1.")
        elif n is not None and frac is None and n % 1 != 0:
            raise ValueError("Only integers accepted as `n` values")
        elif (n is not None) == (frac is not None):
            raise ValueError("Please enter a value for `frac` OR `n`, not both")

        return DataFrame(
            _query_compiler=self._query_compiler.sample(
                n=n, frac=frac, random_state=random_state
            )
        )

    def drop(
        self,
        labels=None,
        axis=0,
        index=None,
        columns=None,
        level=None,
        inplace=False,
        errors="raise",
    ):
        """Return new object with labels in requested axis removed.

        Parameters
        ----------
        labels:
            Index or column labels to drop.
        axis:
            Whether to drop labels from the index (0 / 'index') or columns (1 / 'columns').
        index, columns:
            Alternative to specifying axis (labels, axis=1 is equivalent to columns=labels).
        level:
            For MultiIndex - not supported
        inplace:
            If True, do operation inplace and return None.
        errors:
            If 'ignore', suppress error and existing labels are dropped.

        Returns
        -------
        dropped:
            type of caller

        See Also
        --------
        :pandas_api_docs:`pandas.DataFrame.drop`

        Examples
        --------
        Drop a column

        >>> df = ed.DataFrame('http://localhost:9200', 'ecommerce', columns=['customer_first_name', 'email', 'user'])
        >>> df.drop(columns=['user'])
             customer_first_name                       email
        0                  Eddie  eddie@underwood-family.zzz
        1                   Mary      mary@bailey-family.zzz
        2                   Gwen      gwen@butler-family.zzz
        3                  Diane   diane@chandler-family.zzz
        4                  Eddie      eddie@weber-family.zzz
        ...                  ...                         ...
        4670                Mary     mary@lambert-family.zzz
        4671                 Jim      jim@gilbert-family.zzz
        4672               Yahya     yahya@rivera-family.zzz
        4673                Mary     mary@hampton-family.zzz
        4674             Jackson  jackson@hopkins-family.zzz
        <BLANKLINE>
        [4675 rows x 2 columns]

        Drop rows by index value (axis=0)

        >>> df.drop(['1', '2'])
             customer_first_name                       email     user
        0                  Eddie  eddie@underwood-family.zzz    eddie
        3                  Diane   diane@chandler-family.zzz    diane
        4                  Eddie      eddie@weber-family.zzz    eddie
        5                  Diane    diane@goodwin-family.zzz    diane
        6                 Oliver      oliver@rios-family.zzz   oliver
        ...                  ...                         ...      ...
        4670                Mary     mary@lambert-family.zzz     mary
        4671                 Jim      jim@gilbert-family.zzz      jim
        4672               Yahya     yahya@rivera-family.zzz    yahya
        4673                Mary     mary@hampton-family.zzz     mary
        4674             Jackson  jackson@hopkins-family.zzz  jackson
        <BLANKLINE>
        [4673 rows x 3 columns]
        """
        # Level not supported
        if level is not None:
            raise NotImplementedError(f"level not supported {level}")

        inplace = validate_bool_kwarg(inplace, "inplace")
        if labels is not None:
            if index is not None or columns is not None:
                raise ValueError("Cannot specify both 'labels' and 'index'/'columns'")
            axis = pd.DataFrame._get_axis_name(axis)
            axes = {axis: labels}
        elif index is not None or columns is not None:
            axes, _ = pd.DataFrame()._construct_axes_from_arguments(
                (index, columns), {}
            )
        else:
            raise ValueError(
                "Need to specify at least one of 'labels', 'index' or 'columns'"
            )

        # TODO Clean up this error checking
        if "index" not in axes:
            axes["index"] = None
        elif axes["index"] is not None:
            if not is_list_like(axes["index"]):
                axes["index"] = [axes["index"]]
            if errors == "raise":
                # Check if axes['index'] values exists in index
                count = self._query_compiler._index_matches_count(axes["index"])
                if count != len(axes["index"]):
                    raise ValueError(
                        f"number of labels {count}!={len(axes['index'])} not contained in axis"
                    )
            else:
                """
                axes["index"] = self._query_compiler.index_matches(axes["index"])
                # If the length is zero, we will just do nothing
                if not len(axes["index"]):
                    axes["index"] = None
                """
                raise NotImplementedError()

        if "columns" not in axes:
            axes["columns"] = None
        elif axes["columns"] is not None:
            if not is_list_like(axes["columns"]):
                axes["columns"] = [axes["columns"]]
            if errors == "raise":
                non_existant = [
                    obj for obj in axes["columns"] if obj not in self.columns
                ]
                if len(non_existant):
                    raise ValueError(f"labels {non_existant} not contained in axis")
            else:
                axes["columns"] = [
                    obj for obj in axes["columns"] if obj in self.columns
                ]
                # If the length is zero, we will just do nothing
                if not len(axes["columns"]):
                    axes["columns"] = None

        new_query_compiler = self._query_compiler.drop(
            index=axes["index"], columns=axes["columns"]
        )
        return self._create_or_update_from_compiler(new_query_compiler, inplace)

    def __getitem__(self, key):
        return self._getitem(key)

    def __dir__(self):
        """
        Provide autocompletion on field names in interactive environment.
        """
        return super().__dir__() + [
            column_name
            for column_name in self._query_compiler.columns.to_list()
            if is_valid_attr_name(column_name)
        ]

    def __repr__(self) -> None:
        """
        From pandas
        """
        buf = StringIO()

        # max_rows and max_cols determine the maximum size of the pretty printed tabular
        # representation of the dataframe. pandas defaults are 60 and 20 respectively.
        # dataframes where len(df) > max_rows shows a truncated view with 10 rows shown.
        max_rows = pd.get_option("display.max_rows")
        max_cols = pd.get_option("display.max_columns")
        min_rows = pd.get_option("display.min_rows")

        if max_rows and len(self) > max_rows:
            max_rows = min_rows

        show_dimensions = pd.get_option("display.show_dimensions")
        if pd.get_option("display.expand_frame_repr"):
            width, _ = console.get_console_size()
        else:
            width = None

        self.to_string(
            buf=buf,
            max_rows=max_rows,
            max_cols=max_cols,
            line_width=width,
            show_dimensions=show_dimensions,
        )

        return buf.getvalue()

    def _info_repr(self) -> bool:
        """
        True if the repr should show the info view.
        """
        info_repr_option = pd.get_option("display.large_repr") == "info"
        return info_repr_option and not (
            self._repr_fits_horizontal_() and self._repr_fits_vertical_()
        )

    def _repr_html_(self) -> Optional[str]:
        """
        From pandas - this is called by notebooks
        """
        if self._info_repr():
            buf = StringIO("")
            self.info(buf=buf)
            # need to escape the <class>, should be the first line.
            val = buf.getvalue().replace("<", r"&lt;", 1)
            val = val.replace(">", r"&gt;", 1)
            return "<pre>" + val + "</pre>"

        if pd.get_option("display.notebook_repr_html"):
            max_rows = pd.get_option("display.max_rows")
            max_cols = pd.get_option("display.max_columns")
            min_rows = pd.get_option("display.min_rows")
            show_dimensions = pd.get_option("display.show_dimensions")

            if max_rows and len(self) > max_rows:
                max_rows = min_rows

            return self.to_html(
                max_rows=max_rows,
                max_cols=max_cols,
                show_dimensions=show_dimensions,
                notebook=True,
            )  # set for consistency with pandas output
        else:
            return None

    def count(self) -> pd.Series:
        """
        Count non-NA cells for each column.

        Counts are based on exists queries against ES.

        This is inefficient, as it creates N queries (N is number of fields).
        An alternative approach is to use value_count aggregations. However, they have issues in that:

        - They can only be used with aggregatable fields (e.g. keyword not text)
        - For list fields they return multiple counts. E.g. tags=['elastic', 'ml'] returns value_count=2 for a
          single document.

        TODO - add additional pandas.DataFrame.count features

        Returns
        -------
        pandas.Series:
            Summary of column counts

        See Also
        --------
        :pandas_api_docs:`pandas.DataFrame.count`

        Examples
        --------
        >>> df = ed.DataFrame('http://localhost:9200', 'ecommerce', columns=['customer_first_name', 'geoip.city_name'])
        >>> df.count()
        customer_first_name    4675
        geoip.city_name        4094
        dtype: int64
        """
        return self._query_compiler.count()

    def es_info(self):
        # noinspection PyPep8
        """
        A debug summary of an eland DataFrame internals.

        This includes the Elasticsearch search queries and query compiler task list.

        Returns
        -------
        str
            A debug summary of an eland DataFrame internals.

        Examples
        --------
        >>> df = ed.DataFrame('http://localhost:9200', 'flights')
        >>> df = df[(df.OriginAirportID == 'AMS') & (df.FlightDelayMin > 60)]
        >>> df = df[['timestamp', 'OriginAirportID', 'DestAirportID', 'FlightDelayMin']]
        >>> df = df.tail()
        >>> df
                        timestamp OriginAirportID DestAirportID  FlightDelayMin
        12608 2018-02-10 01:20:52             AMS          CYEG             120
        12720 2018-02-10 14:09:40             AMS           BHM             255
        12725 2018-02-10 00:53:01             AMS           ATL             360
        12823 2018-02-10 15:41:20             AMS           NGO             120
        12907 2018-02-11 20:08:25             AMS           LIM             225
        <BLANKLINE>
        [5 rows x 4 columns]
        >>> print(df.es_info())
        es_index_pattern: flights
        Index:
         es_index_field: _id
         is_source_field: False
        Mappings:
         capabilities:
                           es_field_name  is_source es_dtype                  es_date_format        pd_dtype  is_searchable  is_aggregatable  is_scripted aggregatable_es_field_name
        timestamp              timestamp       True     date  strict_date_hour_minute_second  datetime64[ns]           True             True        False                  timestamp
        OriginAirportID  OriginAirportID       True  keyword                            None          object           True             True        False            OriginAirportID
        DestAirportID      DestAirportID       True  keyword                            None          object           True             True        False              DestAirportID
        FlightDelayMin    FlightDelayMin       True  integer                            None           int64           True             True        False             FlightDelayMin
        Operations:
         tasks: [('boolean_filter': ('boolean_filter': {'bool': {'must': [{'term': {'OriginAirportID': 'AMS'}}, {'range': {'FlightDelayMin': {'gt': 60}}}]}})), ('tail': ('sort_field': '_doc', 'count': 5))]
         size: 5
         sort_params: {'_doc': 'desc'}
         _source: ['timestamp', 'OriginAirportID', 'DestAirportID', 'FlightDelayMin']
         body: {'query': {'bool': {'must': [{'term': {'OriginAirportID': 'AMS'}}, {'range': {'FlightDelayMin': {'gt': 60}}}]}}}
         post_processing: [('sort_index')]
        <BLANKLINE>
        """
        buf = StringIO()

        super()._es_info(buf)

        return buf.getvalue()

    def es_match(
        self,
        text: str,
        *,
        columns: Optional[Union[str, Sequence[str]]] = None,
        match_phrase: bool = False,
        must_not_match: bool = False,
        multi_match_type: Optional[str] = None,
        match_only_text_fields: bool = True,
        analyzer: Optional[str] = None,
        fuzziness: Optional[Union[int, str]] = None,
        **kwargs: Any,
    ) -> "DataFrame":
        """Filters data with an Elasticsearch ``match``, ``match_phrase``, or
        ``multi_match`` query depending on the given parameters and columns.

        Read more about `Full-Text Queries in Elasticsearch <https://www.elastic.co/guide/en/elasticsearch/reference/current/full-text-queries.html>`_

        By default all fields of type 'text' within Elasticsearch are queried
        otherwise specific columns can be specified via the ``columns`` parameter
        or a single column can be filtered on with :py:meth:`eland.Series.es_match`

        All additional keyword arguments are passed in the body of the match query.

        Parameters
        ----------
        text: str
            String of text to search for
        columns: str, List[str], optional
            List of columns to search over. Defaults to all 'text' fields in Elasticsearch
        match_phrase: bool, default False
            If True will use ``match_phrase`` instead of ``match`` query which takes into account
            the order of the ``text`` parameter.
        must_not_match: bool, default False
            If True will apply a boolean NOT (~) to the
            query. Instead of requiring a match the query
            will require text to not match.
        multi_match_type: str, optional
            If given and matching against multiple columns will set the ``multi_match.type`` setting
        match_only_text_fields: bool, default True
            When True this function will raise an error if any non-text fields
            are queried to prevent fields that aren't analyzed from not working properly.
            Set to False to ignore this preventative check.
        analyzer: str, optional
            Specify which analyzer to use for the match query
        fuzziness: int, str, optional
            Specify the fuzziness option for the match query

        Returns
        -------
        DataFrame
            A filtered :py:class:`eland.DataFrame` with the given match query

        Examples
        --------
        >>> df = ed.DataFrame("http://localhost:9200", "ecommerce")
        >>> df.es_match("Men's", columns=["category"])
                                                      category currency  ...   type     user
        0                                     [Men's Clothing]      EUR  ...  order    eddie
        4                  [Men's Clothing, Men's Accessories]      EUR  ...  order    eddie
        6                                     [Men's Clothing]      EUR  ...  order   oliver
        7     [Men's Clothing, Men's Accessories, Men's Shoes]      EUR  ...  order      abd
        11                 [Men's Accessories, Men's Clothing]      EUR  ...  order    eddie
        ...                                                ...      ...  ...    ...      ...
        4663                     [Men's Shoes, Men's Clothing]      EUR  ...  order    samir
        4667                     [Men's Clothing, Men's Shoes]      EUR  ...  order   sultan
        4671                                  [Men's Clothing]      EUR  ...  order      jim
        4672                                  [Men's Clothing]      EUR  ...  order    yahya
        4674             [Women's Accessories, Men's Clothing]      EUR  ...  order  jackson
        <BLANKLINE>
        [2310 rows x 45 columns]
        """
        # Determine which columns will be used
        es_dtypes = self.es_dtypes.to_dict()
        if columns is None:
            columns = [
                column for column, es_dtype in es_dtypes.items() if es_dtype == "text"
            ]
        elif isinstance(columns, str):
            columns = [columns]
        columns = list(columns)

        qc = self._query_compiler
        filter = qc.es_match(
            text,
            columns,
            match_phrase=match_phrase,
            match_only_text_fields=match_only_text_fields,
            multi_match_type=multi_match_type,
            analyzer=analyzer,
            fuzziness=fuzziness,
            **kwargs,
        )
        if must_not_match:
            filter = ~filter
        return DataFrame(_query_compiler=qc._update_query(filter))

    def es_query(self, query) -> "DataFrame":
        """Applies an Elasticsearch DSL query to the current DataFrame.

        Parameters
        ----------
        query:
            Dictionary of the Elasticsearch DSL query to apply

        Returns
        -------
        eland.DataFrame:
            eland DataFrame with the query applied

        Examples
        --------

        Apply a `geo-distance query`_ to a dataset with a geo-point column ``geoip.location``.

         .. _geo-distance query: https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-geo-distance-query.html

        >>> df = ed.DataFrame('http://localhost:9200', 'ecommerce', columns=['customer_first_name', 'geoip.city_name'])
        >>> df.es_query({"bool": {"filter": {"geo_distance": {"distance": "1km", "geoip.location": [55.3, 25.3]}}}}).head()
           customer_first_name geoip.city_name
        1                 Mary           Dubai
        9            Rabbia Al           Dubai
        10           Rabbia Al           Dubai
        22                Mary           Dubai
        30              Robbie           Dubai
        <BLANKLINE>
        [5 rows x 2 columns]

        If using an occurrence like ``must`` or ``filter`` you must
        nest it within ``bool``:

         .. code-block:: python

            # Correct:
            df.es_query({
                "bool": {
                    "filter": {...}
                }
            })

            # Incorrect, needs to be nested under 'bool':
            df.es_query({
                "filter": {...}
            })
        """
        # Unpack the {'query': ...} which some
        # users may use due to documentation.
        if not isinstance(query, dict):
            raise TypeError("'query' must be of type 'dict'")
        if tuple(query) == ("query",):
            query = query["query"]
        return DataFrame(_query_compiler=self._query_compiler.es_query(query))

    def _index_summary(self):
        # Print index summary e.g.
        # Index: 103 entries, 0 to 102
        # Do this by getting head and tail of dataframe
        if self.empty:
            # index[0] is out of bounds for empty df
            head = self.head(1).to_pandas()
            tail = self.tail(1).to_pandas()
        else:
            head = self.head(1).to_pandas().index[0]
            tail = self.tail(1).to_pandas().index[0]
        index_summary = f", {pprint_thing(head)} to {pprint_thing(tail)}"

        name = "Index"
        return f"{name}: {len(self)} entries{index_summary}"

    def info(
        self,
        verbose: Optional[bool] = None,
        buf: Optional[StringIO] = None,
        max_cols: Optional[int] = None,
        memory_usage: Optional[bool] = None,
        show_counts: Optional[bool] = None,
    ) -> None:
        """
        Print a concise summary of a DataFrame.

        This method prints information about a DataFrame including
        the index dtype and column dtypes, non-null values and memory usage.

        See :pandas_api_docs:`pandas.DataFrame.info` for full details.

        Notes
        -----
        This copies a lot of code from pandas.DataFrame.info as it is difficult
        to split out the appropriate code or creating a SparseDataFrame gives
        incorrect results on types and counts.

        Examples
        --------
        >>> df = ed.DataFrame('http://localhost:9200', 'ecommerce', columns=['customer_first_name', 'geoip.city_name'])
        >>> df.info()
        <class 'eland.dataframe.DataFrame'>
        Index: 4675 entries, 0 to 4674
        Data columns (total 2 columns):
         #   Column               Non-Null Count  Dtype...
        ---  ------               --------------  -----...
         0   customer_first_name  4675 non-null   object
         1   geoip.city_name      4094 non-null   object
        dtypes: object(2)
        memory usage: ...
        Elasticsearch storage usage: ...
        """
        if buf is None:  # pragma: no cover
            buf = sys.stdout

        lines = [str(type(self)), self._index_summary()]

        columns: pd.Index = self.columns
        number_of_columns: int = len(columns)

        if number_of_columns == 0:
            lines.append(f"Empty {type(self).__name__}")
            fmt.buffer_put_lines(buf, lines)
            return

        # hack
        if max_cols is None:
            max_cols = pd.get_option("display.max_info_columns", number_of_columns + 1)

        max_rows = pd.get_option("display.max_info_rows", len(self) + 1)

        if show_counts is None:
            show_counts = (number_of_columns <= max_cols) and (len(self) < max_rows)

        exceeds_info_cols = number_of_columns > max_cols

        # From pandas.DataFrame
        def _put_str(s, space) -> str:
            return f"{s}"[:space].ljust(space)

        def _verbose_repr(number_of_columns: int) -> None:
            lines.append(f"Data columns (total {number_of_columns} columns):")

            id_head = " # "
            column_head = "Column"
            col_space = 2

            max_col = max(len(pprint_thing(k)) for k in columns)
            len_column = len(pprint_thing(column_head))
            space = max(max_col, len_column) + col_space

            max_id = len(pprint_thing(number_of_columns))
            len_id = len(pprint_thing(id_head))
            space_num = max(max_id, len_id) + col_space
            counts = None

            header = _put_str(id_head, space_num) + _put_str(column_head, space)
            if show_counts:
                counts = self.count()
                if number_of_columns != len(counts):  # pragma: no cover
                    raise AssertionError(
                        f"Columns must equal counts ({number_of_columns:d} != {len(counts):d})"
                    )
                count_header = "Non-Null Count"
                len_count = len(count_header)
                non_null = " non-null"
                max_count = max(len(pprint_thing(k)) for k in counts) + len(non_null)
                space_count = max(len_count, max_count) + col_space
                count_temp = "{count}" + non_null
            else:
                count_header = ""
                space_count = len(count_header)
                len_count = space_count
                count_temp = "{count}"

            dtype_header = "Dtype"
            len_dtype = len(dtype_header)
            max_dtypes = max(len(pprint_thing(k)) for k in self.dtypes)
            space_dtype = max(len_dtype, max_dtypes)
            header += _put_str(count_header, space_count) + _put_str(
                dtype_header, space_dtype
            )

            lines.append(header)
            lines.append(
                _put_str("-" * len_id, space_num)
                + _put_str("-" * len_column, space)
                + _put_str("-" * len_count, space_count)
                + _put_str("-" * len_dtype, space_dtype)
            )

            dtypes = self.dtypes
            for i, col in enumerate(columns):
                dtype = dtypes.iloc[i]
                col = pprint_thing(col)

                line_no = _put_str(f" {i}", space_num)

                count = ""
                if show_counts:
                    count = counts.iloc[i]

                lines.append(
                    line_no
                    + _put_str(col, space)
                    + _put_str(count_temp.format(count=count), space_count)
                    + _put_str(dtype, space_dtype)
                )

        def _non_verbose_repr() -> None:
            lines.append(columns._summary(name="Columns"))

        def _sizeof_fmt(num: float, size_qualifier: str) -> str:
            # returns size in human readable format
            for x in ["bytes", "KB", "MB", "GB", "TB"]:
                if num < 1024.0:
                    return f"{num:3.3f}{size_qualifier} {x}"
                num /= 1024.0
            return f"{num:3.3f}{size_qualifier} PB"

        if verbose:
            _verbose_repr(number_of_columns)
        elif verbose is False:  # specifically set to False, not nesc None
            _non_verbose_repr()
        else:
            _non_verbose_repr() if exceeds_info_cols else _verbose_repr(
                number_of_columns
            )

        # pandas 0.25.1 uses get_dtype_counts() here. This
        # returns a Series with strings as the index NOT dtypes.
        # Therefore, to get consistent ordering we need to
        # align types with pandas method.
        counts = self.dtypes.value_counts()
        counts.index = counts.index.astype(str)
        dtypes = [f"{k}({v:d})" for k, v in sorted(counts.items())]
        lines.append(f"dtypes: {', '.join(dtypes)}")

        if memory_usage is None:
            memory_usage = pd.get_option("display.memory_usage")
        if memory_usage:
            # append memory usage of df to display
            size_qualifier = ""

            # TODO - this is different from pd.DataFrame as we shouldn't
            #   really hold much in memory. For now just approximate with getsizeof + ignore deep
            mem_usage = sys.getsizeof(self)
            lines.append(f"memory usage: {_sizeof_fmt(mem_usage, size_qualifier)}")
            storage_usage = self._query_compiler._client.indices.stats(
                index=self._query_compiler._index_pattern, metric=["store"]
            )["_all"]["total"]["store"]["size_in_bytes"]
            lines.append(
                f"Elasticsearch storage usage: {_sizeof_fmt(storage_usage,size_qualifier)}\n"
            )

        fmt.buffer_put_lines(buf, lines)

    @docstring_parameter(DEFAULT_NUM_ROWS_DISPLAYED)
    def to_html(
        self,
        buf=None,
        columns=None,
        col_space=None,
        header=True,
        index=True,
        na_rep="NaN",
        formatters=None,
        float_format=None,
        sparsify=None,
        index_names=True,
        justify=None,
        max_rows=None,
        max_cols=None,
        show_dimensions=False,
        decimal=".",
        bold_rows=True,
        classes=None,
        escape=True,
        notebook=False,
        border=None,
        table_id=None,
        render_links=False,
    ) -> Any:
        """
        Render a Elasticsearch data as an HTML table.

        Follows pandas implementation except when ``max_rows=None``. In this scenario, we set ``max_rows={0}`` to avoid
        accidentally dumping an entire index. This can be overridden by explicitly setting ``max_rows``.

        See Also
        --------
        :pandas_api_docs:`pandas.DataFrame.to_html`
        """
        # In pandas calling 'to_string' without max_rows set, will dump ALL rows - we avoid this
        # by limiting rows by default.
        num_rows = len(self)  # avoid multiple calls
        if num_rows <= DEFAULT_NUM_ROWS_DISPLAYED:
            if max_rows is None:
                max_rows = num_rows
            else:
                max_rows = min(num_rows, max_rows)
        elif max_rows is None:
            warnings.warn(
                f"DataFrame.to_string called without max_rows set "
                f"- this will return entire index results. "
                f"Setting max_rows={DEFAULT_NUM_ROWS_DISPLAYED}"
                f" overwrite if different behaviour is required.",
                UserWarning,
            )
            max_rows = DEFAULT_NUM_ROWS_DISPLAYED

        # because of the way pandas handles max_rows=0, not having this throws an error
        # see eland issue #56
        if max_rows == 0:
            max_rows = 1

        # Create a slightly bigger dataframe than display
        df = self._build_repr(max_rows + 1)

        if buf is not None:
            _buf = _expand_user(stringify_path(buf))
        else:
            _buf = StringIO()

        df.to_html(
            buf=_buf,
            columns=columns,
            col_space=col_space,
            header=header,
            index=index,
            na_rep=na_rep,
            formatters=formatters,
            float_format=float_format,
            sparsify=sparsify,
            index_names=index_names,
            justify=justify,
            max_rows=max_rows,
            max_cols=max_cols,
            show_dimensions=False,
            decimal=decimal,
            bold_rows=bold_rows,
            classes=classes,
            escape=escape,
            notebook=notebook,
            border=border,
            table_id=table_id,
            render_links=render_links,
        )

        # Our fake dataframe has incorrect number of rows (max_rows*2+1) - write out
        # the correct number of rows
        if show_dimensions:
            # TODO - this results in different output to pandas
            # TODO - the 'x' character is different and this gets added after the </div>
            _buf.write(f"\n<p>{len(self.index)} rows × {len(self.columns)} columns</p>")

        if buf is None:
            result = _buf.getvalue()
            return result

    @docstring_parameter(DEFAULT_NUM_ROWS_DISPLAYED)
    def to_string(
        self,
        buf=None,
        columns=None,
        col_space=None,
        header=True,
        index=True,
        na_rep="NaN",
        formatters=None,
        float_format=None,
        sparsify=None,
        index_names=True,
        justify=None,
        max_rows=None,
        max_cols=None,
        show_dimensions=False,
        decimal=".",
        line_width=None,
    ):
        """
        Render a DataFrame to a console-friendly tabular output.

        Follows pandas implementation except when ``max_rows=None``. In this scenario, we set ``max_rows={0}`` to avoid
        accidentally dumping an entire index. This can be overridden by explicitly setting ``max_rows``.

        See Also
        --------
        :pandas_api_docs:`pandas.DataFrame.to_string`
        """
        # In pandas calling 'to_string' without max_rows set, will dump ALL rows - we avoid this
        # by limiting rows by default.
        num_rows = len(self)  # avoid multiple calls
        if num_rows <= DEFAULT_NUM_ROWS_DISPLAYED:
            if max_rows is None:
                max_rows = num_rows
            else:
                max_rows = min(num_rows, max_rows)
        elif max_rows is None:
            warnings.warn(
                f"DataFrame.to_string called without max_rows set "
                f"- this will return entire index results. "
                f"Setting max_rows={DEFAULT_NUM_ROWS_DISPLAYED}"
                f" overwrite if different behaviour is required.",
                UserWarning,
            )
            max_rows = DEFAULT_NUM_ROWS_DISPLAYED

        # because of the way pandas handles max_rows=0, not having this throws an error
        # see eland issue #56
        if max_rows == 0:
            max_rows = 1

        # Create a slightly bigger dataframe than display
        df = self._build_repr(max_rows + 1)

        if buf is not None:
            _buf = _expand_user(stringify_path(buf))
        else:
            _buf = StringIO()

        df.to_string(
            buf=_buf,
            columns=columns,
            col_space=col_space,
            na_rep=na_rep,
            formatters=formatters,
            float_format=float_format,
            sparsify=sparsify,
            justify=justify,
            index_names=index_names,
            header=header,
            index=index,
            max_rows=max_rows,
            max_cols=max_cols,
            show_dimensions=False,  # print this outside of this call
            decimal=decimal,
            line_width=line_width,
        )

        # Our fake dataframe has incorrect number of rows (max_rows*2+1) - write out
        # the correct number of rows
        if show_dimensions:
            _buf.write(f"\n\n[{len(self.index)} rows x {len(self.columns)} columns]")

        if buf is None:
            result = _buf.getvalue()
            return result

    def __getattr__(self, key: str) -> Any:
        """After regular attribute access, looks up the name in the columns

        Parameters
        ----------
            key: str
                Attribute name.

        Returns
        -------
            The value of the attribute.
        """
        try:
            return object.__getattribute__(self, key)
        except AttributeError as e:
            if key in self.columns:
                return self[key]
            raise e

    def _getitem(
        self,
        key: Union[
            "DataFrame", "Series", pd.Index, List[str], str, BooleanFilter, np.ndarray
        ],
    ) -> Union["Series", "DataFrame"]:
        """Get the column specified by key for this DataFrame.

        Args:
            key : The column name.

        Returns:
            A Pandas Series representing the value for the column.
        """
        key = apply_if_callable(key, self)
        # Shortcut if key is an actual column
        try:
            if key in self.columns:
                return self._getitem_column(key)
        except (KeyError, ValueError, TypeError):
            pass
        if isinstance(key, (Series, np.ndarray, pd.Index, list)):
            return self._getitem_array(key)
        elif isinstance(key, DataFrame):
            return self.where(key)
        elif isinstance(key, BooleanFilter):
            return DataFrame(_query_compiler=self._query_compiler._update_query(key))
        else:
            return self._getitem_column(key)

    def _getitem_column(self, key: str) -> "Series":
        if key not in self.columns:
            raise KeyError(f"Requested column [{key}] is not in the DataFrame.")
        s = self._reduce_dimension(self._query_compiler.getitem_column_array([key]))
        return s

    def _getitem_array(self, key: Union[str, pd.Series]) -> "DataFrame":
        if isinstance(key, Series):
            key = key.to_pandas()
        if is_bool_indexer(key):
            if isinstance(key, pd.Series) and not key.index.equals(self.index):
                warnings.warn(
                    "Boolean Series key will be reindexed to match DataFrame index.",
                    PendingDeprecationWarning,
                    stacklevel=3,
                )
            elif len(key) != len(self.index):
                raise ValueError(
                    f"Item wrong length {len(key)} instead of {len(self.index)}."
                )
            key = check_bool_indexer(self.index, key)
            # We convert to a RangeIndex because getitem_row_array is expecting a list
            # of indices, and RangeIndex will give us the exact indices of each boolean
            # requested.
            key = pd.RangeIndex(len(self.index))[key]
            if len(key):
                return DataFrame(
                    _query_compiler=self._query_compiler.getitem_row_array(key)
                )
            else:
                return DataFrame(columns=self.columns)
        else:
            if any(k not in self.columns for k in key):
                raise KeyError(
                    f"{str([k for k in key if k not in self.columns]).replace(',', '')}"
                    f" not index"
                )
            return DataFrame(
                _query_compiler=self._query_compiler.getitem_column_array(key)
            )

    def _create_or_update_from_compiler(
        self, new_query_compiler: "QueryCompiler", inplace: bool = False
    ) -> Union["QueryCompiler", "DataFrame"]:
        """Returns or updates a DataFrame given new query_compiler"""
        assert (
            isinstance(new_query_compiler, type(self._query_compiler))
            or type(new_query_compiler) in self._query_compiler.__class__.__bases__
        ), f"Invalid Query Compiler object: {type(new_query_compiler)}"
        if not inplace:
            return DataFrame(_query_compiler=new_query_compiler)
        else:
            self._query_compiler: "QueryCompiler" = new_query_compiler

    @staticmethod
    def _reduce_dimension(query_compiler: "QueryCompiler") -> "Series":
        return Series(_query_compiler=query_compiler)

    def to_csv(
        self,
        path_or_buf=None,
        sep=",",
        na_rep="",
        float_format=None,
        columns=None,
        header=True,
        index=True,
        index_label=None,
        mode="w",
        encoding=None,
        compression="infer",
        quoting=None,
        quotechar='"',
        line_terminator=None,
        chunksize=None,
        tupleize_cols=None,
        date_format=None,
        doublequote=True,
        escapechar=None,
        decimal=".",
    ) -> Optional[str]:
        """
        Write Elasticsearch data to a comma-separated values (csv) file.

        See Also
        --------
        :pandas_api_docs:`pandas.DataFrame.to_csv`
        """
        kwargs = {
            "path_or_buf": path_or_buf,
            "sep": sep,
            "na_rep": na_rep,
            "float_format": float_format,
            "columns": columns,
            "header": header,
            "index": index,
            "index_label": index_label,
            "mode": mode,
            "encoding": encoding,
            "compression": compression,
            "quoting": quoting,
            "quotechar": quotechar,
            "line_terminator": line_terminator,
            "chunksize": chunksize,
            "date_format": date_format,
            "doublequote": doublequote,
            "escapechar": escapechar,
            "decimal": decimal,
        }
        return self._query_compiler.to_csv(**kwargs)

    def to_pandas(self, show_progress: bool = False) -> pd.DataFrame:
        """
        Utility method to convert eland.Dataframe to pandas.Dataframe

        Returns
        -------
        pandas.DataFrame
        """
        return self._query_compiler.to_pandas(show_progress=show_progress)

    def _empty_pd_df(self) -> pd.DataFrame:
        return self._query_compiler._empty_pd_ef()

    def select_dtypes(self, include=None, exclude=None) -> "DataFrame":
        """
        Return a subset of the DataFrame's columns based on the column dtypes.

        Compatible with :pandas_api_docs:`pandas.DataFrame.select_dtypes`

        Returns
        -------
        eland.DataFrame
            DataFrame contains only columns of selected dtypes

        Examples
        --------
        >>> df = ed.DataFrame('http://localhost:9200', 'flights',
        ... columns=['AvgTicketPrice', 'Dest', 'Cancelled', 'timestamp', 'dayOfWeek'])
        >>> df.dtypes
        AvgTicketPrice           float64
        Dest                      object
        Cancelled                   bool
        timestamp         datetime64[ns]
        dayOfWeek                  int64
        dtype: object
        >>> df = df.select_dtypes(include=[np.number, 'datetime'])
        >>> df.dtypes
        AvgTicketPrice           float64
        timestamp         datetime64[ns]
        dayOfWeek                  int64
        dtype: object
        """
        empty_df = self._empty_pd_df()

        empty_df = empty_df.select_dtypes(include=include, exclude=exclude)

        return self._getitem_array(empty_df.columns)

    @property
    def shape(self) -> Tuple[int, int]:
        """
        Return a tuple representing the dimensionality of the DataFrame.

        Returns
        -------
        shape: tuple

        0. number of rows
        1. number of columns

        Notes
        -----
        - number of rows ``len(df)`` queries Elasticsearch
        - number of columns ``len(df.columns)`` is cached. If mappings are updated, DataFrame must be updated.

        Examples
        --------
        >>> df = ed.DataFrame('http://localhost:9200', 'ecommerce')
        >>> df.shape
        (4675, 45)
        """
        num_rows = len(self)
        num_columns = len(self.columns)

        return num_rows, num_columns

    @property
    def ndim(self) -> int:
        """
        Returns 2 by definition of a DataFrame

        Returns
        -------
        int
            By definition 2

        See Also
        --------
        :pandas_api_docs:`pandas.DataFrame.ndim`
        """
        return 2

    def keys(self) -> pd.Index:
        """
        Return columns

        See :pandas_api_docs:`pandas.DataFrame.keys`

        Returns
        -------
        pandas.Index
            Elasticsearch field names as pandas.Index
        """
        return self.columns

    def iterrows(self) -> Iterable[Tuple[Union[str, Tuple[str, ...]], pd.Series]]:
        """
        Iterate over eland.DataFrame rows as (index, pandas.Series) pairs.

        Yields
        ------
        index: index
            The index of the row.
        data: pandas Series
            The data of the row as a pandas Series.

        See Also
        --------
        eland.DataFrame.itertuples: Iterate over eland.DataFrame rows as namedtuples.

        Examples
        --------
        >>> df = ed.DataFrame('http://localhost:9200', 'flights', columns=['AvgTicketPrice', 'Cancelled']).head()
        >>> df
           AvgTicketPrice  Cancelled
        0      841.265642      False
        1      882.982662      False
        2      190.636904      False
        3      181.694216       True
        4      730.041778      False
        <BLANKLINE>
        [5 rows x 2 columns]

        >>> for index, row in df.iterrows():
        ...     print(row)
        AvgTicketPrice    841.265642
        Cancelled              False
        Name: 0, dtype: object
        AvgTicketPrice    882.982662
        Cancelled              False
        Name: 1, dtype: object
        AvgTicketPrice    190.636904
        Cancelled              False
        Name: 2, dtype: object
        AvgTicketPrice    181.694216
        Cancelled               True
        Name: 3, dtype: object
        AvgTicketPrice    730.041778
        Cancelled              False
        Name: 4, dtype: object
        """
        for df in self._query_compiler.search_yield_pandas_dataframes():
            yield from df.iterrows()

    def itertuples(
        self, index: bool = True, name: Union[str, None] = "Eland"
    ) -> Iterable[Tuple[Any, ...]]:
        """
        Iterate over eland.DataFrame rows as namedtuples.

        Parameters
        ----------
        index: bool, default True
            If True, return the index as the first element of the tuple.
        name: str or None, default "Eland"
            The name of the returned namedtuples or None to return regular tuples.

        Returns
        -------
        iterator
            An object to iterate over namedtuples for each row in the
            DataFrame with the first field possibly being the index and
            following fields being the column values.

        See Also
        --------
        eland.DataFrame.iterrows: Iterate over eland.DataFrame rows as (index, pandas.Series) pairs.

        Examples
        --------
        >>> df = ed.DataFrame('http://localhost:9200', 'flights', columns=['AvgTicketPrice', 'Cancelled']).head()
        >>> df
           AvgTicketPrice  Cancelled
        0      841.265642      False
        1      882.982662      False
        2      190.636904      False
        3      181.694216       True
        4      730.041778      False
        <BLANKLINE>
        [5 rows x 2 columns]

        >>> for row in df.itertuples():
        ...     print(row)
        Eland(Index='0', AvgTicketPrice=841.2656419677076, Cancelled=False)
        Eland(Index='1', AvgTicketPrice=882.9826615595518, Cancelled=False)
        Eland(Index='2', AvgTicketPrice=190.6369038508356, Cancelled=False)
        Eland(Index='3', AvgTicketPrice=181.69421554118, Cancelled=True)
        Eland(Index='4', AvgTicketPrice=730.041778346198, Cancelled=False)

        By setting the `index` parameter to False we can remove the index as the first element of the tuple:

        >>> for row in df.itertuples(index=False):
        ...     print(row)
        Eland(AvgTicketPrice=841.2656419677076, Cancelled=False)
        Eland(AvgTicketPrice=882.9826615595518, Cancelled=False)
        Eland(AvgTicketPrice=190.6369038508356, Cancelled=False)
        Eland(AvgTicketPrice=181.69421554118, Cancelled=True)
        Eland(AvgTicketPrice=730.041778346198, Cancelled=False)

        With the `name` parameter set we set a custom name for the yielded namedtuples:

        >>> for row in df.itertuples(name='Flight'):
        ...     print(row)
        Flight(Index='0', AvgTicketPrice=841.2656419677076, Cancelled=False)
        Flight(Index='1', AvgTicketPrice=882.9826615595518, Cancelled=False)
        Flight(Index='2', AvgTicketPrice=190.6369038508356, Cancelled=False)
        Flight(Index='3', AvgTicketPrice=181.69421554118, Cancelled=True)
        Flight(Index='4', AvgTicketPrice=730.041778346198, Cancelled=False)
        """
        for df in self._query_compiler.search_yield_pandas_dataframes():
            yield from df.itertuples(index=index, name=name)

    def aggregate(
        self,
        func: Union[str, List[str]],
        axis: int = 0,
        numeric_only: Optional[bool] = None,
        *args,
        **kwargs,
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Aggregate using one or more operations over the specified axis.

        Parameters
        ----------
        func: function, str, list or dict
            Function to use for aggregating the data. If a function, must either
            work when passed a %(klass)s or when passed to %(klass)s.apply.

            Accepted combinations are:

            - function
            - string function name
            - list of functions and/or function names, e.g. ``[np.sum, 'mean']``
            - dict of axis labels -> functions, function names or list of such.

            Currently, we only support ``['count', 'mad', 'max', 'mean', 'median', 'min', 'mode', 'quantile',
            'rank', 'sem', 'skew', 'sum', 'std', 'var']``
        axis: int
            Currently, we only support axis=0 (index)
        numeric_only: {True, False, None} Default is None
            Which datatype to be returned
            - True: returns all values with float64, NaN/NaT are ignored.
            - False: returns all values with float64.
            - None: returns all values with default datatype.
        *args
            Positional arguments to pass to `func`
        **kwargs
            Keyword arguments to pass to `func`

        Returns
        -------
        DataFrame, Series or scalar
            if DataFrame.agg is called with a single function, returns a Series
            if DataFrame.agg is called with several functions, returns a DataFrame
            if Series.agg is called with single function, returns a scalar
            if Series.agg is called with several functions, returns a Series

        See Also
        --------
        :pandas_api_docs:`pandas.DataFrame.aggregate`

        Examples
        --------
        >>> df = ed.DataFrame('http://localhost:9200', 'flights', columns=['AvgTicketPrice', 'DistanceKilometers', 'timestamp', 'DestCountry'])
        >>> df.aggregate(['sum', 'min', 'std'], numeric_only=True).astype(int)
             AvgTicketPrice  DistanceKilometers
        sum         8204364            92616288
        min             100                   0
        std             266                4578

        >>> df.aggregate(['sum', 'min', 'std'], numeric_only=True)
             AvgTicketPrice  DistanceKilometers
        sum    8.204365e+06        9.261629e+07
        min    1.000205e+02        0.000000e+00
        std    2.664071e+02        4.578614e+03

        >>> df.aggregate(['sum', 'min', 'std'], numeric_only=False)
             AvgTicketPrice  DistanceKilometers  timestamp  DestCountry
        sum    8.204365e+06        9.261629e+07        NaT          NaN
        min    1.000205e+02        0.000000e+00 2018-01-01          NaN
        std    2.664071e+02        4.578614e+03        NaT          NaN

        >>> df.aggregate(['sum', 'min', 'std'], numeric_only=None)
             AvgTicketPrice  DistanceKilometers  timestamp  DestCountry
        sum    8.204365e+06        9.261629e+07        NaT          NaN
        min    1.000205e+02        0.000000e+00 2018-01-01          NaN
        std    2.664071e+02        4.578614e+03        NaT          NaN
        """
        axis = pd.DataFrame._get_axis_number(axis)

        if axis == 1:
            raise NotImplementedError(
                "Aggregating via index not currently implemented - needs index transform"
            )

        # currently we only support a subset of functions that aggregate columns.
        # ['count', 'mad', 'max', 'mean', 'median', 'min', 'mode', 'quantile',
        # 'rank', 'sem', 'skew', 'sum', 'std', 'var', 'nunique']
        if isinstance(func, str):
            # Wrap in list
            return (
                self._query_compiler.aggs([func], numeric_only=numeric_only)
                .squeeze()
                .rename(None)
            )
        elif is_list_like(func):
            # we have a list!
            return self._query_compiler.aggs(func, numeric_only=numeric_only)

    agg = aggregate

    hist = gfx.ed_hist_frame

    def groupby(
        self, by: Optional[Union[str, List[str]]] = None, dropna: bool = True
    ) -> "DataFrameGroupBy":
        """
        Used to perform groupby operations

        Parameters
        ----------
        by:
            column or list of columns used to groupby
            Currently accepts column or list of columns

        dropna: default True
            If True, and if group keys contain NA values, NA values together with row/column will be dropped.

        Returns
        -------
        eland.groupby.DataFrameGroupBy

        See Also
        --------
        :pandas_api_docs:`pandas.DataFrame.groupby`

        Examples
        --------
        >>> ed_flights = ed.DataFrame('http://localhost:9200', 'flights', columns=["AvgTicketPrice", "Cancelled", "dayOfWeek", "timestamp", "DestCountry"])
        >>> ed_flights.groupby(["DestCountry", "Cancelled"]).agg(["min", "max"], numeric_only=True) # doctest: +NORMALIZE_WHITESPACE
                              AvgTicketPrice              dayOfWeek
                                         min          max       min  max
        DestCountry Cancelled
        AE          False         110.799911  1126.148682       0.0  6.0
                    True          132.443756   817.931030       0.0  6.0
        AR          False         125.589394  1199.642822       0.0  6.0
                    True          251.389603  1172.382568       0.0  6.0
        AT          False         100.020531  1181.835815       0.0  6.0
        ...                              ...          ...       ...  ...
        TR          True          307.915649   307.915649       0.0  0.0
        US          False         100.145966  1199.729004       0.0  6.0
                    True          102.153069  1192.429932       0.0  6.0
        ZA          False         102.002663  1196.186157       0.0  6.0
                    True          121.280296  1175.709961       0.0  6.0
        <BLANKLINE>
        [63 rows x 4 columns]

        >>> ed_flights.groupby(["DestCountry", "Cancelled"]).mean(numeric_only=True) # doctest: +NORMALIZE_WHITESPACE
                               AvgTicketPrice  dayOfWeek
        DestCountry Cancelled
        AE          False          643.956793   2.717949
                    True           388.828809   2.571429
        AR          False          673.551677   2.746154
                    True           682.197241   2.733333
        AT          False          647.158290   2.819936
        ...                               ...        ...
        TR          True           307.915649   0.000000
        US          False          598.063146   2.752014
                    True           579.799066   2.767068
        ZA          False          636.998605   2.738589
                    True           677.794078   2.928571
        <BLANKLINE>
        [63 rows x 2 columns]

        >>> ed_flights.groupby(["DestCountry", "Cancelled"]).min(numeric_only=False) # doctest: +NORMALIZE_WHITESPACE
                               AvgTicketPrice  dayOfWeek           timestamp
        DestCountry Cancelled
        AE          False          110.799911          0 2018-01-01 19:31:30
                    True           132.443756          0 2018-01-06 13:03:25
        AR          False          125.589394          0 2018-01-01 01:30:47
                    True           251.389603          0 2018-01-01 02:13:17
        AT          False          100.020531          0 2018-01-01 05:24:19
        ...                               ...        ...                 ...
        TR          True           307.915649          0 2018-01-08 04:35:10
        US          False          100.145966          0 2018-01-01 00:06:27
                    True           102.153069          0 2018-01-01 09:02:36
        ZA          False          102.002663          0 2018-01-01 06:44:44
                    True           121.280296          0 2018-01-04 00:37:01
        <BLANKLINE>
        [63 rows x 3 columns]
        """
        if by is None:
            raise ValueError("by parameter should be specified to groupby")
        if isinstance(by, str):
            by = [by]
        if isinstance(by, (list, tuple)):
            remaining_columns = sorted(set(by) - set(self._query_compiler.columns))
            if remaining_columns:
                raise KeyError(
                    f"Requested columns {repr(remaining_columns)[1:-1]} not in the DataFrame"
                )

        return DataFrameGroupBy(
            by=by, query_compiler=self._query_compiler.copy(), dropna=dropna
        )

    def mode(
        self,
        numeric_only: bool = False,
        dropna: bool = True,
        es_size: int = 10,
    ) -> pd.DataFrame:
        """
        Calculate mode of a DataFrame

        Parameters
        ----------
        numeric_only: {True, False} Default is False
            Which datatype to be returned
            - True: Returns all numeric or timestamp columns
            - False: Returns all columns
        dropna: {True, False} Default is True
            - True: Don’t consider counts of NaN/NaT.
            - False: Consider counts of NaN/NaT.
        es_size: default 10
            number of rows to be returned if mode has multiple values

        See Also
        --------
        :pandas_api_docs:`pandas.DataFrame.mode`

        Examples
        --------
        >>> ed_ecommerce = ed.DataFrame('http://localhost:9200', 'ecommerce')
        >>> ed_df = ed_ecommerce.filter(["total_quantity", "geoip.city_name", "customer_birth_date", "day_of_week", "taxful_total_price"])
        >>> ed_df.mode(numeric_only=False)
           total_quantity geoip.city_name customer_birth_date day_of_week  taxful_total_price
        0               2        New York                 NaT    Thursday               53.98

        >>> ed_df.mode(numeric_only=True)
           total_quantity  taxful_total_price
        0               2               53.98

        >>> ed_df = ed_ecommerce.filter(["products.tax_amount","order_date"])
        >>> ed_df.mode()
           products.tax_amount          order_date
        0                  0.0 2016-12-02 20:36:58
        1                  NaN 2016-12-04 23:44:10
        2                  NaN 2016-12-08 06:21:36
        3                  NaN 2016-12-08 09:38:53
        4                  NaN 2016-12-12 11:38:24
        5                  NaN 2016-12-12 19:46:34
        6                  NaN 2016-12-14 18:00:00
        7                  NaN 2016-12-15 11:38:24
        8                  NaN 2016-12-22 19:39:22
        9                  NaN 2016-12-24 06:21:36

        >>> ed_df.mode(es_size = 3)
           products.tax_amount          order_date
        0                  0.0 2016-12-02 20:36:58
        1                  NaN 2016-12-04 23:44:10
        2                  NaN 2016-12-08 06:21:36
        """
        # TODO dropna=False
        return self._query_compiler.mode(
            numeric_only=numeric_only, dropna=True, is_dataframe=True, es_size=es_size
        )

    def quantile(
        self,
        q: Union[int, float, List[int], List[float]] = 0.5,
        numeric_only: Optional[bool] = True,
    ) -> "pd.DataFrame":
        """
        Used to calculate quantile for a given DataFrame.

        Parameters
        ----------
        q:
            float or array like, default 0.5
            Value between 0 <= q <= 1, the quantile(s) to compute.
        numeric_only: {True, False, None} Default is True
            Which datatype to be returned
            - True: Returns all values as float64, NaN/NaT values are removed
            - None: Returns all values as the same dtype where possible, NaN/NaT are removed
            - False: Returns all values as the same dtype where possible, NaN/NaT are preserved

        Returns
        -------
        pandas.DataFrame
            quantile value for each column

        See Also
        --------
        :pandas_api_docs:`pandas.DataFrame.quantile`

        Examples
        --------
        >>> ed_df = ed.DataFrame('http://localhost:9200', 'flights')
        >>> ed_flights = ed_df.filter(["AvgTicketPrice", "FlightDelayMin", "dayOfWeek", "timestamp"])
        >>> ed_flights.quantile() # doctest: +SKIP
        AvgTicketPrice    640.387285
        FlightDelayMin      0.000000
        dayOfWeek           3.000000
        Name: 0.5, dtype: float64

        >>> ed_flights.quantile([.2, .5, .75]) # doctest: +SKIP
              AvgTicketPrice  FlightDelayMin  dayOfWeek
        0.20      361.040768             0.0        1.0
        0.50      640.387285             0.0        3.0
        0.75      842.213490            15.0        4.0

        >>> ed_flights.quantile([.2, .5, .75], numeric_only=False) # doctest: +SKIP
              AvgTicketPrice  FlightDelayMin  dayOfWeek                     timestamp
        0.20      361.040768             0.0        1.0 2018-01-09 04:43:55.296587520
        0.50      640.387285             0.0        3.0 2018-01-21 23:51:57.637076736
        0.75      842.213490            15.0        4.0 2018-02-01 04:46:16.658119680
        """
        return self._query_compiler.quantile(quantiles=q, numeric_only=numeric_only)

    def idxmax(self, axis: int = 0) -> pd.Series:
        """
        Return index of first occurrence of maximum over requested axis.

        NA/null values are excluded.

        Parameters
        ----------
        axis : {0, 1}, default 0
            The axis to filter on, expressed as index (int).

        Returns
        -------
        pandas.Series

        See Also
        --------
        :pandas_api_docs:`pandas.DataFrame.idxmax`

        Examples
        --------
        >>> ed_df = ed.DataFrame('http://localhost:9200', 'flights')
        >>> ed_flights = ed_df.filter(["AvgTicketPrice", "FlightDelayMin", "dayOfWeek", "timestamp"])
        >>> ed_flights.idxmax()
        AvgTicketPrice    1843
        FlightDelayMin     109
        dayOfWeek         1988
        dtype: object

        """
        return self._query_compiler.idx(axis=axis, sort_order="desc")

    def idxmin(self, axis: int = 0) -> pd.Series:
        """
        Return index of first occurrence of minimum over requested axis.

        NA/null values are excluded.

        Parameters
        ----------
        axis : {0, 1}, default 0
            The axis to filter on, expressed as index (int).

        Returns
        -------
        pandas.Series

        See Also
        --------
        :pandas_api_docs:`pandas.DataFrame.idxmin`

        Examples
        --------
        >>> ed_df = ed.DataFrame('http://localhost:9200', 'flights')
        >>> ed_flights = ed_df.filter(["AvgTicketPrice", "FlightDelayMin", "dayOfWeek", "timestamp"])
        >>> ed_flights.idxmin()
        AvgTicketPrice    5454
        FlightDelayMin       0
        dayOfWeek            0
        dtype: object

        """
        return self._query_compiler.idx(axis=axis, sort_order="asc")

    def query(self, expr) -> "DataFrame":
        """
        Query the columns of a DataFrame with a boolean expression.

        TODO - add additional pandas arguments

        Parameters
        ----------
        expr: str
            A boolean expression

        Returns
        -------
        eland.DataFrame:
            DataFrame populated by results of the query

        TODO - add link to eland user guide

        See Also
        --------
        :pandas_api_docs:`pandas.DataFrame.query`
        :pandas_user_guide:`indexing`

        Examples
        --------
        >>> df = ed.DataFrame('http://localhost:9200', 'flights')
        >>> df.shape
        (13059, 27)
        >>> df.query('FlightDelayMin > 60').shape
        (2730, 27)
        """
        if isinstance(expr, BooleanFilter):
            return DataFrame(
                _query_compiler=self._query_compiler._update_query(BooleanFilter(expr))
            )
        elif isinstance(expr, str):
            column_resolver = {}
            for key in self.keys():
                column_resolver[key] = self.get(key)
            # Create fake resolvers - index resolver is empty
            resolvers = column_resolver, {}
            # Use pandas eval to parse query - TODO validate this further
            filter = eval(expr, target=self, resolvers=tuple(tuple(resolvers)))
            return DataFrame(_query_compiler=self._query_compiler._update_query(filter))
        else:
            raise NotImplementedError(expr, type(expr))

    def get(
        self, key: Any, default: Optional[Any] = None
    ) -> Union["Series", "DataFrame"]:
        """
        Get item from object for given key (ex: DataFrame column).
        Returns default value if not found.

        Parameters
        ----------
        key: object
        default: default value if not found

        Returns
        -------
        value: same type as items contained in object

        See Also
        --------
        :pandas_api_docs:`pandas.DataFrame.get`

        Examples
        --------
        >>> df = ed.DataFrame('http://localhost:9200', 'flights')
        >>> df.get('Carrier')
        0         Kibana Airlines
        1        Logstash Airways
        2        Logstash Airways
        3         Kibana Airlines
        4         Kibana Airlines
                       ...
        13054    Logstash Airways
        13055    Logstash Airways
        13056    Logstash Airways
        13057            JetBeats
        13058            JetBeats
        Name: Carrier, Length: 13059, dtype: object
        """
        if key in self.keys():
            return self._getitem(key)
        else:
            return default

    def filter(
        self,
        items: Optional[Sequence[str]] = None,
        like: Optional[str] = None,
        regex: Optional[str] = None,
        axis: Optional[Union[int, str]] = None,
    ) -> "DataFrame":
        """
        Subset the dataframe rows or columns according to the specified index labels.
        Note that this routine does not filter a dataframe on its
        contents. The filter is applied to the labels of the index.

        Parameters
        ----------
        items : list-like
            Keep labels from axis which are in items.
        like : str
            Keep labels from axis for which "like in label == True".
        regex : str (regular expression)
            Keep labels from axis for which re.search(regex, label) == True.
        axis : {0 or ‘index’, 1 or ‘columns’, None}, default None
            The axis to filter on, expressed either as an index (int) or axis name (str). By default this is the info axis, ‘index’ for Series, ‘columns’ for DataFrame.

        Returns
        -------
        eland.DataFrame

        See Also
        --------
        :pandas_api_docs:`pandas.DataFrame.filter`

        Notes
        -----
        The ``items``, ``like``, and ``regex`` parameters are
        enforced to be mutually exclusive.
        """
        filter_options_passed = sum([items is not None, bool(like), bool(regex)])
        if filter_options_passed > 1:
            raise TypeError(
                "Keyword arguments `items`, `like`, or `regex` "
                "are mutually exclusive"
            )
        elif filter_options_passed == 0:
            raise TypeError("Must pass either 'items', 'like', or 'regex'")

        # axis defaults to 'columns' for DataFrame, 'index' for Series
        if axis is None:
            axis = "columns"
        axis = pd.DataFrame._get_axis_name(axis)

        if axis == "index":
            new_query_compiler = self._query_compiler.filter(
                items=items, like=like, regex=regex
            )
            return self._create_or_update_from_compiler(
                new_query_compiler, inplace=False
            )

        else:  # axis == "columns"
            if items is not None:
                # Pandas skips over columns that don't exist
                # and maintains order of items=[...]
                existing_columns = set(self.columns.to_list())
                return self[[column for column in items if column in existing_columns]]

            elif like is not None:

                def matcher(x: str) -> bool:
                    return like in x

            else:
                matcher = re.compile(regex).search

            return self[[column for column in self.columns if matcher(column)]]

    @property
    def values(self) -> None:
        """
        Not implemented.

        In pandas this returns a Numpy representation of the DataFrame. This would involve scan/scrolling the
        entire index.

        If this is required, call ``ed.eland_to_pandas(ed_df).values``, *but beware this will scan/scroll the entire
        Elasticsearch index(s) into memory.*

        See Also
        --------
        :pandas_api_docs:`pandas.DataFrame.values`
        eland_to_pandas
        to_numpy
        """
        return self.to_numpy()

    def to_numpy(self) -> None:
        """
        Not implemented.

        In pandas this returns a Numpy representation of the DataFrame. This would involve scan/scrolling the
        entire index.

        If this is required, call ``ed.eland_to_pandas(ed_df).values``, *but beware this will scan/scroll the entire
        Elasticsearch index(s) into memory.*

        See Also
        --------
        :pandas_api_docs:`pandas.DataFrame.to_numpy`
        eland_to_pandas

        Examples
        --------
        >>> ed_df = ed.DataFrame('http://localhost:9200', 'flights', columns=['AvgTicketPrice', 'Carrier']).head(5)
        >>> pd_df = ed.eland_to_pandas(ed_df)
        >>> print(f"type(ed_df)={type(ed_df)}\\ntype(pd_df)={type(pd_df)}")
        type(ed_df)=<class 'eland.dataframe.DataFrame'>
        type(pd_df)=<class 'pandas.core.frame.DataFrame'>
        >>> ed_df
           AvgTicketPrice           Carrier
        0      841.265642   Kibana Airlines
        1      882.982662  Logstash Airways
        2      190.636904  Logstash Airways
        3      181.694216   Kibana Airlines
        4      730.041778   Kibana Airlines
        <BLANKLINE>
        [5 rows x 2 columns]
        >>> pd_df.values
        array([[841.2656419677076, 'Kibana Airlines'],
               [882.9826615595518, 'Logstash Airways'],
               [190.6369038508356, 'Logstash Airways'],
               [181.69421554118, 'Kibana Airlines'],
               [730.041778346198, 'Kibana Airlines']], dtype=object)
        """
        raise AttributeError(
            "This method would scan/scroll the entire Elasticsearch index(s) into memory. "
            "If this is explicitly required, and there is sufficient memory, call `ed.eland_to_pandas(ed_df).values`"
        )
