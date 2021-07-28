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

"""
Series
---------
One-dimensional ndarray with axis labels (including time series).

The underlying data resides in Elasticsearch and the API aligns as much as
possible with pandas.DataFrame API.

This allows the eland.Series to access large datasets stored in Elasticsearch,
without storing the dataset in local memory.

Implementation Details
----------------------
Based on NDFrame which underpins eland.DataFrame
"""

import sys
import warnings
from collections.abc import Collection
from io import StringIO
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd  # type: ignore
from pandas.io.common import _expand_user, stringify_path  # type: ignore

import eland.plotting
from eland.arithmetics import ArithmeticNumber, ArithmeticSeries, ArithmeticString
from eland.common import DEFAULT_NUM_ROWS_DISPLAYED, docstring_parameter
from eland.filter import (
    BooleanFilter,
    Equal,
    Greater,
    GreaterEqual,
    IsIn,
    IsNull,
    Less,
    LessEqual,
    NotFilter,
    NotNull,
    QueryFilter,
    ScriptFilter,
)
from eland.ndframe import NDFrame
from eland.utils import to_list

if TYPE_CHECKING:
    from elasticsearch import Elasticsearch

    from eland.query_compiler import QueryCompiler


def _get_method_name() -> str:
    return sys._getframe(1).f_code.co_name


class Series(NDFrame):
    """
    pandas.Series like API that proxies into Elasticsearch index(es).

    Parameters
    ----------
    es_client : elasticsearch.Elasticsearch
        A reference to a Elasticsearch python client

    es_index_pattern : str
        An Elasticsearch index pattern. This can contain wildcards.

    es_index_field : str
        The field to base the series on

    Notes
    -----
    If the Elasticsearch index is deleted or index mappings are changed after this
    object is created, the object is not rebuilt and so inconsistencies can occur.

    See Also
    --------
    :pandas_api_docs:`pandas.Series`

    Examples
    --------
    >>> ed.Series(es_client='localhost', es_index_pattern='flights', name='Carrier')
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

    def __init__(
        self,
        es_client: Optional["Elasticsearch"] = None,
        es_index_pattern: Optional[str] = None,
        name: Optional[str] = None,
        es_index_field: Optional[str] = None,
        _query_compiler: Optional["QueryCompiler"] = None,
    ) -> None:
        # Series has 1 column
        if name is None:
            columns = None
        else:
            columns = [name]

        super().__init__(
            es_client=es_client,
            es_index_pattern=es_index_pattern,
            columns=columns,
            es_index_field=es_index_field,
            _query_compiler=_query_compiler,
        )

    hist = eland.plotting.ed_hist_series

    @property
    def empty(self) -> bool:
        """Determines if the Series is empty.

        Returns:
            True if the Series is empty.
            False otherwise.
        """
        return len(self.index) == 0

    @property
    def shape(self) -> Tuple[int, int]:
        """
        Return a tuple representing the dimensionality of the Series.

        Returns
        -------
        shape: tuple

        0. number of rows
        1. number of columns

        Notes
        -----
        - number of rows ``len(series)`` queries Elasticsearch
        - number of columns == 1

        Examples
        --------
        >>> df = ed.Series('localhost', 'ecommerce', name='total_quantity')
        >>> df.shape
        (4675, 1)
        """
        num_rows = len(self)
        num_columns = 1

        return num_rows, num_columns

    @property
    def es_field_name(self) -> pd.Index:
        """
        Returns
        -------
        es_field_name: str
            Return the Elasticsearch field name for this series
        """
        return self._query_compiler.get_field_names(include_scripted_fields=True)[0]

    @property
    def name(self) -> pd.Index:
        return self._query_compiler.columns[0]

    @name.setter
    def name(self, name: str) -> None:
        self._query_compiler.rename({self.name: name}, inplace=True)

    def rename(self, new_name: str) -> "Series":
        """
        Rename name of series. Only column rename is supported. This does not change the underlying
        Elasticsearch index, but adds a symbolic link from the new name (column) to the Elasticsearch field name.

        For instance, if a field was called 'total_quantity' it could be renamed 'Total Quantity'.

        Parameters
        ----------
        new_name: str

        Returns
        -------
        eland.Series
            eland.Series with new name.

        See Also
        --------
        :pandas_api_docs:`pandas.Series.rename`

        Examples
        --------
        >>> df = ed.DataFrame('localhost', 'flights')
        >>> df.Carrier
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
        >>> df.Carrier.rename('Airline')
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
        Name: Airline, Length: 13059, dtype: object
        """
        return Series(
            _query_compiler=self._query_compiler.rename({self.name: new_name})
        )

    def head(self, n: int = 5) -> "Series":
        return Series(_query_compiler=self._query_compiler.head(n))

    def tail(self, n: int = 5) -> "Series":
        return Series(_query_compiler=self._query_compiler.tail(n))

    def sample(
        self,
        n: Optional[int] = None,
        frac: Optional[float] = None,
        random_state: Optional[int] = None,
    ) -> "Series":
        return Series(
            _query_compiler=self._query_compiler.sample(n, frac, random_state)
        )

    def value_counts(self, es_size: int = 10) -> pd.Series:
        """
        Return the value counts for the specified field.

        **Note we can only do this for aggregatable Elasticsearch fields - (in general) numeric and keyword
        rather than text fields**

        TODO - implement remainder of pandas arguments

        Parameters
        ----------
        es_size: int, default 10
            Number of buckets to return counts for, automatically sorts by count descending.
            This parameter is specific to `eland`, and determines how many term buckets
            elasticsearch should return out of the overall terms list.

        Returns
        -------
        pandas.Series
            number of occurrences of each value in the column

        See Also
        --------
        :pandas_api_docs:`pandas.Series.value_counts`
        :es_api_docs:`search-aggregations-bucket-terms-aggregation`

        Examples
        --------
        >>> df = ed.DataFrame('localhost', 'flights')
        >>> df['Carrier'].value_counts()
        Logstash Airways    3331
        JetBeats            3274
        Kibana Airlines     3234
        ES-Air              3220
        Name: Carrier, dtype: int64
        """
        if not isinstance(es_size, int):
            raise TypeError("es_size must be a positive integer.")
        elif es_size <= 0:
            raise ValueError("es_size must be a positive integer.")
        return self._query_compiler.value_counts(es_size)

    # dtype not implemented for Series as causes query to fail
    # in pandas.core.computation.ops.Term.type

    # ----------------------------------------------------------------------
    # Rendering Methods
    def __repr__(self) -> str:
        """
        Return a string representation for a particular Series.
        """
        buf = StringIO()

        # max_rows and max_cols determine the maximum size of the pretty printed tabular
        # representation of the series. pandas defaults are 60 and 20 respectively.
        # series where len(series) > max_rows shows a truncated view with 10 rows shown.
        max_rows = pd.get_option("display.max_rows")
        min_rows = pd.get_option("display.min_rows")

        if max_rows and len(self) > max_rows:
            max_rows = min_rows

        show_dimensions = pd.get_option("display.show_dimensions")

        self.to_string(
            buf=buf,
            name=True,
            dtype=True,
            min_rows=min_rows,
            max_rows=max_rows,
            length=show_dimensions,
        )
        result = buf.getvalue()

        return result

    @docstring_parameter(DEFAULT_NUM_ROWS_DISPLAYED)
    def to_string(
        self,
        buf=None,
        na_rep="NaN",
        float_format=None,
        header=True,
        index=True,
        length=False,
        dtype=False,
        name=False,
        max_rows=None,
        min_rows=None,
    ) -> Optional[str]:
        """
        Render a string representation of the Series.

        Follows pandas implementation except when ``max_rows=None``. In this scenario, we set ``max_rows={0}`` to avoid
        accidentally dumping an entire index. This can be overridden by explicitly setting ``max_rows``.

        See Also
        --------
        :pandas_api_docs:`pandas.Series.to_string`
            for argument details.
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
                f"Series.to_string called without max_rows set "
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
        temp_series = self._build_repr(max_rows + 1)

        if buf is not None:
            _buf = _expand_user(stringify_path(buf))
        else:
            _buf = StringIO()

        if num_rows == 0:
            # Empty series are rendered differently than
            # series with items. We can luckily use our
            # example series in this case.
            temp_series.head(0).to_string(
                buf=_buf,
                na_rep=na_rep,
                float_format=float_format,
                header=header,
                index=index,
                length=length,
                dtype=dtype,
                name=name,
                max_rows=max_rows,
            )
        else:
            # Create repr of fake series without name, length, dtype summary
            temp_series.to_string(
                buf=_buf,
                na_rep=na_rep,
                float_format=float_format,
                header=header,
                index=index,
                length=False,
                dtype=False,
                name=False,
                max_rows=max_rows,
            )

            # Create the summary
            footer = []
            if name and self.name is not None:
                footer.append(f"Name: {self.name}")
            if length and len(self) > max_rows:
                footer.append(f"Length: {len(self.index)}")
            if dtype:
                footer.append(f"dtype: {temp_series.dtype}")

            if footer:
                _buf.write(f"\n{', '.join(footer)}")

        if buf is None:
            result = _buf.getvalue()
            return result

    def to_pandas(self, show_progress: bool = False) -> pd.Series:
        return self._query_compiler.to_pandas(show_progress=show_progress)[self.name]

    @property
    def dtype(self) -> np.dtype:
        """
        Return the dtype object of the underlying data.

        See Also
        --------
        :pandas_api_docs:`pandas.Series.dtype`
        """
        return self._query_compiler.dtypes[0]

    @property
    def es_dtype(self) -> str:
        """
        Return the Elasticsearch type of the underlying data.
        """
        return self._query_compiler.es_dtypes[0]

    def __gt__(self, other: Union[int, float, "Series"]) -> BooleanFilter:
        if isinstance(other, Series):
            # Need to use scripted query to compare to values
            painless = f"doc['{self.name}'].value > doc['{other.name}'].value"
            return ScriptFilter(painless, lang="painless")
        elif isinstance(other, (int, float)):
            return Greater(field=self.name, value=other)
        else:
            raise NotImplementedError(other, type(other))

    def __lt__(self, other: Union[int, float, "Series"]) -> BooleanFilter:
        if isinstance(other, Series):
            # Need to use scripted query to compare to values
            painless = f"doc['{self.name}'].value < doc['{other.name}'].value"
            return ScriptFilter(painless, lang="painless")
        elif isinstance(other, (int, float)):
            return Less(field=self.name, value=other)
        else:
            raise NotImplementedError(other, type(other))

    def __ge__(self, other: Union[int, float, "Series"]) -> BooleanFilter:
        if isinstance(other, Series):
            # Need to use scripted query to compare to values
            painless = f"doc['{self.name}'].value >= doc['{other.name}'].value"
            return ScriptFilter(painless, lang="painless")
        elif isinstance(other, (int, float)):
            return GreaterEqual(field=self.name, value=other)
        else:
            raise NotImplementedError(other, type(other))

    def __le__(self, other: Union[int, float, "Series"]) -> BooleanFilter:
        if isinstance(other, Series):
            # Need to use scripted query to compare to values
            painless = f"doc['{self.name}'].value <= doc['{other.name}'].value"
            return ScriptFilter(painless, lang="painless")
        elif isinstance(other, (int, float)):
            return LessEqual(field=self.name, value=other)
        else:
            raise NotImplementedError(other, type(other))

    def __eq__(self, other: Union[int, float, str, "Series"]) -> BooleanFilter:
        if isinstance(other, Series):
            # Need to use scripted query to compare to values
            painless = f"doc['{self.name}'].value == doc['{other.name}'].value"
            return ScriptFilter(painless, lang="painless")
        elif isinstance(other, (int, float)):
            return Equal(field=self.name, value=other)
        elif isinstance(other, str):
            return Equal(field=self.name, value=other)
        else:
            raise NotImplementedError(other, type(other))

    def __ne__(self, other: Union[int, float, str, "Series"]) -> BooleanFilter:
        if isinstance(other, Series):
            # Need to use scripted query to compare to values
            painless = f"doc['{self.name}'].value != doc['{other.name}'].value"
            return ScriptFilter(painless, lang="painless")
        elif isinstance(other, (int, float)):
            return NotFilter(Equal(field=self.name, value=other))
        elif isinstance(other, str):
            return NotFilter(Equal(field=self.name, value=other))
        else:
            raise NotImplementedError(other, type(other))

    def isin(self, other: Union[Collection, pd.Series]) -> BooleanFilter:
        if isinstance(other, (Collection, pd.Series)):
            return IsIn(field=self.name, value=to_list(other))
        else:
            raise NotImplementedError(other, type(other))

    def isna(self) -> BooleanFilter:
        """
        Detect missing values.

        Returns
        -------
        eland.Series
            Mask of bool values for each element in Series that indicates whether an element is not an NA value.

        See Also
        --------
        :pandas_api_docs:`pandas.Series.isna`
        """
        return IsNull(field=self.name)

    isnull = isna

    def notna(self) -> BooleanFilter:
        """
        Detect existing (non-missing) values.

        Returns
        -------
        eland.Series
            Mask of bool values for each element in Series that indicates whether an element is not an NA value

        See Also
        --------
        :pandas_api_docs:`pandas.Series.notna`

        """
        return NotNull(field=self.name)

    notnull = notna

    def quantile(
        self, q: Union[int, float, List[int], List[float]] = 0.5
    ) -> Union[pd.Series, Any]:
        """
        Used to calculate quantile for a given Series.

        Parameters
        ----------
        q:
            float or array like, default 0.5
            Value between 0 <= q <= 1, the quantile(s) to compute.

        Returns
        -------
        pandas.Series or any single dtype

        See Also
        --------
        :pandas_api_docs:`pandas.Series.quantile`

        Examples
        --------
        >>> ed_flights = ed.DataFrame('localhost', 'flights')
        >>> ed_flights["timestamp"].quantile([.2,.5,.75]) # doctest: +SKIP
        0.20   2018-01-09 04:30:57.289159912
        0.50   2018-01-21 23:39:27.031627441
        0.75   2018-02-01 04:54:59.256136963
        Name: timestamp, dtype: datetime64[ns]

        >>> ed_flights["dayOfWeek"].quantile() # doctest: +SKIP
        3.0

        >>> ed_flights["timestamp"].quantile() # doctest: +SKIP
        Timestamp('2018-01-22 00:12:48.844534180')
        """
        return self._query_compiler.quantile(
            quantiles=q, numeric_only=None, is_dataframe=False
        )

    @property
    def ndim(self) -> int:
        """
        Returns 1 by definition of a Series

        Returns
        -------
        int
            By definition 1

        See Also
        --------
        :pandas_api_docs:`pandas.Series.ndim`
        """
        return 1

    def filter(
        self,
        items: Optional[Sequence[str]] = None,
        like: Optional[str] = None,
        regex: Optional[str] = None,
        axis: Optional[Union[int, str]] = None,
    ) -> "Series":
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
            The axis to filter on, expressed either as an index (int) or axis name (str).
            By default this is the info axis, ‘index’ for Series, ‘columns’ for DataFrame.

        Returns
        -------
        eland.Series

        See Also
        --------
        :pandas_api_docs:`pandas.Series.filter`

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
            axis = "index"
        pd.Series._get_axis_name(axis)

        new_query_compiler = self._query_compiler.filter(
            items=items, like=like, regex=regex
        )
        return Series(_query_compiler=new_query_compiler)

    def mode(self, es_size: int = 10) -> pd.Series:
        """
            Calculate mode of a series

        Parameters
        ----------
        es_size: default 10
            number of rows to be returned if mode has multiple values

        See Also
        --------
        :pandas_api_docs:`pandas.Series.mode`

        Examples
        --------
        >>> ed_ecommerce = ed.DataFrame('localhost', 'ecommerce')
        >>> ed_ecommerce["day_of_week"].mode()
        0    Thursday
        dtype: object

        >>> ed_ecommerce["order_date"].mode()
        0   2016-12-02 20:36:58
        1   2016-12-04 23:44:10
        2   2016-12-08 06:21:36
        3   2016-12-08 09:38:53
        4   2016-12-12 11:38:24
        5   2016-12-12 19:46:34
        6   2016-12-14 18:00:00
        7   2016-12-15 11:38:24
        8   2016-12-22 19:39:22
        9   2016-12-24 06:21:36
        dtype: datetime64[ns]

        >>> ed_ecommerce["order_date"].mode(es_size=3)
        0   2016-12-02 20:36:58
        1   2016-12-04 23:44:10
        2   2016-12-08 06:21:36
        dtype: datetime64[ns]

        """
        return self._query_compiler.mode(is_dataframe=False, es_size=es_size)

    def es_match(
        self,
        text: str,
        *,
        match_phrase: bool = False,
        match_only_text_fields: bool = True,
        analyzer: Optional[str] = None,
        fuzziness: Optional[Union[int, str]] = None,
        **kwargs: Any,
    ) -> QueryFilter:
        """Filters data with an Elasticsearch ``match`` or ``match_phrase``
        query depending on the given parameters.

        Read more about `Full-Text Queries in Elasticsearch <https://www.elastic.co/guide/en/elasticsearch/reference/current/full-text-queries.html>`_

        All additional keyword arguments are passed in the body of the match query.

        Parameters
        ----------
        text: str
            String of text to search for
        match_phrase: bool, default False
            If True will use ``match_phrase`` instead of ``match`` query which takes into account
            the order of the ``text`` parameter.
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
        QueryFilter
            Boolean filter to be combined with other filters and
            then passed to DataFrame[...].

        Examples
        --------
        >>> df = ed.DataFrame(
        ...   "localhost:9200", "ecommerce",
        ...   columns=["category", "taxful_total_price"]
        ... )
        >>> df[
        ...     df.category.es_match("Men's")
        ...     & (df.taxful_total_price > 200.0)
        ... ].head(5)
                                       category  taxful_total_price
        13                     [Men's Clothing]              266.96
        33                     [Men's Clothing]              221.98
        54                     [Men's Clothing]              234.98
        93   [Men's Shoes, Women's Accessories]              239.98
        273                       [Men's Shoes]              214.98
        <BLANKLINE>
        [5 rows x 2 columns]
        """
        return self._query_compiler.es_match(
            text,
            columns=[self.name],
            match_phrase=match_phrase,
            match_only_text_fields=match_only_text_fields,
            analyzer=analyzer,
            fuzziness=fuzziness,
            **kwargs,
        )

    def es_info(self) -> str:
        buf = StringIO()

        super()._es_info(buf)

        return buf.getvalue()

    def __add__(self, right: "Series") -> "Series":
        """
        Return addition of series and right, element-wise (binary operator add).

        Parameters
        ----------
        right: eland.Series

        Returns
        -------
        eland.Series

        Examples
        --------
        >>> df = ed.DataFrame('localhost', 'ecommerce').head(5)
        >>> df.taxful_total_price
        0     36.98
        1     53.98
        2    199.98
        3    174.98
        4     80.98
        Name: taxful_total_price, dtype: float64
        >>> df.taxful_total_price + 1
        0     37.980000
        1     54.980000
        2    200.979996
        3    175.979996
        4     81.980003
        Name: taxful_total_price, dtype: float64
        >>> df.total_quantity
        0    2
        1    2
        2    2
        3    2
        4    2
        Name: total_quantity, dtype: int64
        >>> df.taxful_total_price + df.total_quantity
        0     38.980000
        1     55.980000
        2    201.979996
        3    176.979996
        4     82.980003
        dtype: float64
        >>> df.customer_first_name + df.customer_last_name
        0    EddieUnderwood
        1        MaryBailey
        2        GwenButler
        3     DianeChandler
        4        EddieWeber
        dtype: object
        >>> "First name: " + df.customer_first_name
        0    First name: Eddie
        1     First name: Mary
        2     First name: Gwen
        3    First name: Diane
        4    First name: Eddie
        Name: customer_first_name, dtype: object
        """
        return self._numeric_op(right, _get_method_name())

    def __truediv__(self, right: "Series") -> "Series":
        """
        Return floating division of series and right, element-wise (binary operator truediv).

        Parameters
        ----------
        right: eland.Series

        Returns
        -------
        eland.Series

        Examples
        --------
        >>> df = ed.DataFrame('localhost', 'ecommerce').head(5)
        >>> df.taxful_total_price
        0     36.98
        1     53.98
        2    199.98
        3    174.98
        4     80.98
        Name: taxful_total_price, dtype: float64
        >>> df.total_quantity
        0    2
        1    2
        2    2
        3    2
        4    2
        Name: total_quantity, dtype: int64
        >>> df.taxful_total_price / df.total_quantity
        0    18.490000
        1    26.990000
        2    99.989998
        3    87.489998
        4    40.490002
        dtype: float64
        """
        return self._numeric_op(right, _get_method_name())

    def __floordiv__(self, right: "Series") -> "Series":
        """
        Return integer division of series and right, element-wise (binary operator floordiv //).

        Parameters
        ----------
        right: eland.Series

        Returns
        -------
        eland.Series

        Examples
        --------
        >>> df = ed.DataFrame('localhost', 'ecommerce').head(5)
        >>> df.taxful_total_price
        0     36.98
        1     53.98
        2    199.98
        3    174.98
        4     80.98
        Name: taxful_total_price, dtype: float64
        >>> df.total_quantity
        0    2
        1    2
        2    2
        3    2
        4    2
        Name: total_quantity, dtype: int64
        >>> df.taxful_total_price // df.total_quantity
        0    18.0
        1    26.0
        2    99.0
        3    87.0
        4    40.0
        dtype: float64
        """
        return self._numeric_op(right, _get_method_name())

    def __mod__(self, right: "Series") -> "Series":
        """
        Return modulo of series and right, element-wise (binary operator mod %).

        Parameters
        ----------
        right: eland.Series

        Returns
        -------
        eland.Series

        Examples
        --------
        >>> df = ed.DataFrame('localhost', 'ecommerce').head(5)
        >>> df.taxful_total_price
        0     36.98
        1     53.98
        2    199.98
        3    174.98
        4     80.98
        Name: taxful_total_price, dtype: float64
        >>> df.total_quantity
        0    2
        1    2
        2    2
        3    2
        4    2
        Name: total_quantity, dtype: int64
        >>> df.taxful_total_price % df.total_quantity
        0    0.980000
        1    1.980000
        2    1.979996
        3    0.979996
        4    0.980003
        dtype: float64
        """
        return self._numeric_op(right, _get_method_name())

    def __mul__(self, right: "Series") -> "Series":
        """
        Return multiplication of series and right, element-wise (binary operator mul).

        Parameters
        ----------
        right: eland.Series

        Returns
        -------
        eland.Series

        Examples
        --------
        >>> df = ed.DataFrame('localhost', 'ecommerce').head(5)
        >>> df.taxful_total_price
        0     36.98
        1     53.98
        2    199.98
        3    174.98
        4     80.98
        Name: taxful_total_price, dtype: float64
        >>> df.total_quantity
        0    2
        1    2
        2    2
        3    2
        4    2
        Name: total_quantity, dtype: int64
        >>> df.taxful_total_price * df.total_quantity
        0     73.959999
        1    107.959999
        2    399.959991
        3    349.959991
        4    161.960007
        dtype: float64
        """
        return self._numeric_op(right, _get_method_name())

    def __sub__(self, right: "Series") -> "Series":
        """
        Return subtraction of series and right, element-wise (binary operator sub).

        Parameters
        ----------
        right: eland.Series

        Returns
        -------
        eland.Series

        Examples
        --------
        >>> df = ed.DataFrame('localhost', 'ecommerce').head(5)
        >>> df.taxful_total_price
        0     36.98
        1     53.98
        2    199.98
        3    174.98
        4     80.98
        Name: taxful_total_price, dtype: float64
        >>> df.total_quantity
        0    2
        1    2
        2    2
        3    2
        4    2
        Name: total_quantity, dtype: int64
        >>> df.taxful_total_price - df.total_quantity
        0     34.980000
        1     51.980000
        2    197.979996
        3    172.979996
        4     78.980003
        dtype: float64
        """
        return self._numeric_op(right, _get_method_name())

    def __pow__(self, right: "Series") -> "Series":
        """
        Return exponential power of series and right, element-wise (binary operator pow).

        Parameters
        ----------
        right: eland.Series

        Returns
        -------
        eland.Series

        Examples
        --------
        >>> df = ed.DataFrame('localhost', 'ecommerce').head(5)
        >>> df.taxful_total_price
        0     36.98
        1     53.98
        2    199.98
        3    174.98
        4     80.98
        Name: taxful_total_price, dtype: float64
        >>> df.total_quantity
        0    2
        1    2
        2    2
        3    2
        4    2
        Name: total_quantity, dtype: int64
        >>> df.taxful_total_price ** df.total_quantity
        0     1367.520366
        1     2913.840351
        2    39991.998691
        3    30617.998905
        4     6557.760944
        dtype: float64
        """
        return self._numeric_op(right, _get_method_name())

    def __radd__(self, left: "Series") -> "Series":
        """
        Return addition of series and left, element-wise (binary operator add).

        Parameters
        ----------
        left: eland.Series

        Returns
        -------
        eland.Series

        Examples
        --------
        >>> df = ed.DataFrame('localhost', 'ecommerce').head(5)
        >>> df.taxful_total_price
        0     36.98
        1     53.98
        2    199.98
        3    174.98
        4     80.98
        Name: taxful_total_price, dtype: float64
        >>> 1 + df.taxful_total_price
        0     37.980000
        1     54.980000
        2    200.979996
        3    175.979996
        4     81.980003
        Name: taxful_total_price, dtype: float64
        """
        return self._numeric_op(left, _get_method_name())

    def __rtruediv__(self, left: "Series") -> "Series":
        """
        Return division of series and left, element-wise (binary operator div).

        Parameters
        ----------
        left: eland.Series

        Returns
        -------
        eland.Series

        Examples
        --------
        >>> df = ed.DataFrame('localhost', 'ecommerce').head(5)
        >>> df.taxful_total_price
        0     36.98
        1     53.98
        2    199.98
        3    174.98
        4     80.98
        Name: taxful_total_price, dtype: float64
        >>> 1.0 / df.taxful_total_price
        0    0.027042
        1    0.018525
        2    0.005001
        3    0.005715
        4    0.012349
        Name: taxful_total_price, dtype: float64
        """
        return self._numeric_op(left, _get_method_name())

    def __rfloordiv__(self, left: "Series") -> "Series":
        """
        Return integer division of series and left, element-wise (binary operator floordiv //).

        Parameters
        ----------
        left: eland.Series

        Returns
        -------
        eland.Series

        Examples
        --------
        >>> df = ed.DataFrame('localhost', 'ecommerce').head(5)
        >>> df.taxful_total_price
        0     36.98
        1     53.98
        2    199.98
        3    174.98
        4     80.98
        Name: taxful_total_price, dtype: float64
        >>> 500.0 // df.taxful_total_price
        0    13.0
        1     9.0
        2     2.0
        3     2.0
        4     6.0
        Name: taxful_total_price, dtype: float64
        """
        return self._numeric_op(left, _get_method_name())

    def __rmod__(self, left: "Series") -> "Series":
        """
        Return modulo of series and left, element-wise (binary operator mod %).

        Parameters
        ----------
        left: eland.Series

        Returns
        -------
        eland.Series

        Examples
        --------
        >>> df = ed.DataFrame('localhost', 'ecommerce').head(5)
        >>> df.taxful_total_price
        0     36.98
        1     53.98
        2    199.98
        3    174.98
        4     80.98
        Name: taxful_total_price, dtype: float64
        >>> 500.0 % df.taxful_total_price
        0     19.260006
        1     14.180004
        2    100.040009
        3    150.040009
        4     14.119980
        Name: taxful_total_price, dtype: float64
        """
        return self._numeric_op(left, _get_method_name())

    def __rmul__(self, left: "Series") -> "Series":
        """
        Return multiplication of series and left, element-wise (binary operator mul).

        Parameters
        ----------
        left: eland.Series

        Returns
        -------
        eland.Series

        Examples
        --------
        >>> df = ed.DataFrame('localhost', 'ecommerce').head(5)
        >>> df.taxful_total_price
        0     36.98
        1     53.98
        2    199.98
        3    174.98
        4     80.98
        Name: taxful_total_price, dtype: float64
        >>> 10.0 * df.taxful_total_price
        0     369.799995
        1     539.799995
        2    1999.799957
        3    1749.799957
        4     809.800034
        Name: taxful_total_price, dtype: float64
        """
        return self._numeric_op(left, _get_method_name())

    def __rpow__(self, left: "Series") -> "Series":
        """
        Return exponential power of series and left, element-wise (binary operator pow).

        Parameters
        ----------
        left: eland.Series

        Returns
        -------
        eland.Series

        Examples
        --------
        >>> df = ed.DataFrame('localhost', 'ecommerce').head(5)
        >>> df.total_quantity
        0    2
        1    2
        2    2
        3    2
        4    2
        Name: total_quantity, dtype: int64
        >>> np.int_(2) ** df.total_quantity
        0    4.0
        1    4.0
        2    4.0
        3    4.0
        4    4.0
        Name: total_quantity, dtype: float64
        """
        return self._numeric_op(left, _get_method_name())

    def __rsub__(self, left: "Series") -> "Series":
        """
        Return subtraction of series and left, element-wise (binary operator sub).

        Parameters
        ----------
        left: eland.Series

        Returns
        -------
        eland.Series

        Examples
        --------
        >>> df = ed.DataFrame('localhost', 'ecommerce').head(5)
        >>> df.taxful_total_price
        0     36.98
        1     53.98
        2    199.98
        3    174.98
        4     80.98
        Name: taxful_total_price, dtype: float64
        >>> 1.0 - df.taxful_total_price
        0    -35.980000
        1    -52.980000
        2   -198.979996
        3   -173.979996
        4    -79.980003
        Name: taxful_total_price, dtype: float64
        """
        return self._numeric_op(left, _get_method_name())

    add = __add__
    div = __truediv__
    divide = __truediv__
    floordiv = __floordiv__
    mod = __mod__
    mul = __mul__
    multiply = __mul__
    pow = __pow__
    sub = __sub__
    subtract = __sub__
    truediv = __truediv__

    radd = __radd__
    rdiv = __rtruediv__
    rdivide = __rtruediv__
    rfloordiv = __rfloordiv__
    rmod = __rmod__
    rmul = __rmul__
    rmultiply = __rmul__
    rpow = __rpow__
    rsub = __rsub__
    rsubtract = __rsub__
    rtruediv = __rtruediv__

    # __div__ is technically Python 2.x only
    # but pandas has it so we do too.
    __div__ = __truediv__
    __rdiv__ = __rtruediv__

    def _numeric_op(self, right: Any, method_name: str) -> "Series":
        """
        return a op b

        a & b == Series
            a & b must share same Elasticsearch client, index_pattern and index_field
        a == Series, b == numeric or string

        Naming of the resulting Series
        ------------------------------

        result = SeriesA op SeriesB
        result.name == None

        result = SeriesA op np.number
        result.name == SeriesA.name

        result = SeriesA op str
        result.name == SeriesA.name

        Naming is consistent for rops
        """
        # print("_numeric_op", self, right, method_name)
        if isinstance(right, Series):
            # Check we can the 2 Series are compatible (raises on error):
            self._query_compiler.check_arithmetics(right._query_compiler)

            right_object = ArithmeticSeries(
                right._query_compiler, right.name, right.dtype
            )
            display_name = None
        elif np.issubdtype(np.dtype(type(right)), np.number):
            right_object = ArithmeticNumber(right, np.dtype(type(right)))
            display_name = self.name
        elif isinstance(right, str):
            right_object = ArithmeticString(right)
            display_name = self.name
        else:
            raise TypeError(
                f"unsupported operation type(s) [{method_name!r}] "
                f"for operands ['{type(self)}' with dtype '{self.dtype}', "
                f"'{type(right).__name__}']"
            )

        left_object = ArithmeticSeries(self._query_compiler, self.name, self.dtype)
        left_object.arithmetic_operation(method_name, right_object)

        series = Series(
            _query_compiler=self._query_compiler.arithmetic_op_fields(
                display_name, left_object
            )
        )

        # force set name to 'display_name'
        series._query_compiler._mappings.display_names = [display_name]

        return series

    def max(self, numeric_only: Optional[bool] = None) -> pd.Series:
        """
        Return the maximum of the Series values

        TODO - implement remainder of pandas arguments, currently non-numerics are not supported

        Returns
        -------
        float
            max value

        See Also
        --------
        :pandas_api_docs:`pandas.Series.max`

        Examples
        --------
        >>> s = ed.DataFrame('localhost', 'flights')['AvgTicketPrice']
        >>> int(s.max())
        1199
        """
        results = super().max(numeric_only=numeric_only)
        return results.squeeze()

    def mean(self, numeric_only: Optional[bool] = None) -> pd.Series:
        """
        Return the mean of the Series values

        TODO - implement remainder of pandas arguments, currently non-numerics are not supported

        Returns
        -------
        float
            mean value

        See Also
        --------
        :pandas_api_docs:`pandas.Series.mean`

        Examples
        --------
        >>> s = ed.DataFrame('localhost', 'flights')['AvgTicketPrice']
        >>> int(s.mean())
        628
        """
        results = super().mean(numeric_only=numeric_only)
        return results.squeeze()

    def median(self, numeric_only: Optional[bool] = None) -> pd.Series:
        """
        Return the median of the Series values

        TODO - implement remainder of pandas arguments, currently non-numerics are not supported

        Returns
        -------
        float
            median value

        See Also
        --------
        :pandas_api_docs:`pandas.Series.median`

        Examples
        --------
        >>> s = ed.DataFrame('localhost', 'flights')['AvgTicketPrice']
        >>> int(s.median())
        640
        """
        results = super().median(numeric_only=numeric_only)
        return results.squeeze()

    def min(self, numeric_only: Optional[bool] = None) -> pd.Series:
        """
        Return the minimum of the Series values

        TODO - implement remainder of pandas arguments, currently non-numerics are not supported

        Returns
        -------
        float
            min value

        See Also
        --------
        :pandas_api_docs:`pandas.Series.min`

        Examples
        --------
        >>> s = ed.DataFrame('localhost', 'flights')['AvgTicketPrice']
        >>> int(s.min())
        100
        """
        results = super().min(numeric_only=numeric_only)
        return results.squeeze()

    def sum(self, numeric_only: Optional[bool] = None) -> pd.Series:
        """
        Return the sum of the Series values

        TODO - implement remainder of pandas arguments, currently non-numerics are not supported

        Returns
        -------
        float
            sum of all values

        See Also
        --------
        :pandas_api_docs:`pandas.Series.sum`

        Examples
        --------
        >>> s = ed.DataFrame('localhost', 'flights')['AvgTicketPrice']
        >>> int(s.sum())
        8204364
        """
        results = super().sum(numeric_only=numeric_only)
        return results.squeeze()

    def nunique(self) -> pd.Series:
        """
        Return the number of unique values in a Series

        Returns
        -------
        int
            Number of unique values

        See Also
        --------
        :pandas_api_docs:`pandas.Series.nunique`

        Examples
        --------
        >>> s = ed.DataFrame('localhost', 'flights')['Carrier']
        >>> s.nunique()
        4
        """
        results = super().nunique()
        return results.squeeze()

    def var(self, numeric_only: Optional[bool] = None) -> pd.Series:
        """
        Return variance for a Series

        Returns
        -------
        float
            var value

        See Also
        --------
        :pandas_api_docs:`pandas.Series.var`

        Examples
        --------
        >>> s = ed.DataFrame('localhost', 'flights')['AvgTicketPrice']
        >>> int(s.var())
        70964
        """
        results = super().var(numeric_only=numeric_only)
        return results.squeeze()

    def std(self, numeric_only: Optional[bool] = None) -> pd.Series:
        """
        Return standard deviation for a Series

        Returns
        -------
        float
            std value

        See Also
        --------
        :pandas_api_docs:`pandas.Series.var`

        Examples
        --------
        >>> s = ed.DataFrame('localhost', 'flights')['AvgTicketPrice']
        >>> int(s.std())
        266
        """
        results = super().std(numeric_only=numeric_only)
        return results.squeeze()

    def mad(self, numeric_only: Optional[bool] = None) -> pd.Series:
        """
        Return median absolute deviation for a Series

        Returns
        -------
        float
            mad value

        See Also
        --------
        :pandas_api_docs:`pandas.Series.mad`

        Examples
        --------
        >>> s = ed.DataFrame('localhost', 'flights')['AvgTicketPrice']
        >>> int(s.mad())
        213
        """
        results = super().mad(numeric_only=numeric_only)
        return results.squeeze()

    def describe(self) -> pd.Series:
        """
        Generate descriptive statistics that summarize the central tendency, dispersion and shape of a
        dataset’s distribution, excluding NaN values.

        Analyzes both numeric and object series, as well as DataFrame column sets of mixed data types.
        The output will vary depending on what is provided. Refer to the notes below for more detail.

        TODO - add additional arguments (current only numeric values supported)

        Returns
        -------
        pandas.Series:
            Summary information

        See Also
        --------
        :pandas_api_docs:`pandas.Series.describe`

        Examples
        --------
        >>> df = ed.DataFrame('localhost', 'flights') # ignoring percentiles as they don't generate consistent results
        >>> df.AvgTicketPrice.describe()  # doctest: +SKIP
        count    13059.000000
        mean       628.253689
        std        266.386661
        min        100.020531
        ...
        ...
        ...
        max       1199.729004
        Name: AvgTicketPrice, dtype: float64
        """
        return super().describe().squeeze()

    # def values TODO - not implemented as causes current implementation of query to fail

    def to_numpy(self) -> None:
        """
        Not implemented.

        In pandas this returns a Numpy representation of the Series. This would involve scan/scrolling the
        entire index.

        If this is required, call ``ed.eland_to_pandas(ed_series).values``, *but beware this will scan/scroll the entire
        Elasticsearch index(s) into memory.*

        See Also
        --------
        :pandas_api_docs:`pandas.DataFrame.to_numpy`
        eland_to_pandas

        Examples
        --------
        >>> ed_s = ed.Series('localhost', 'flights', name='Carrier').head(5)
        >>> pd_s = ed.eland_to_pandas(ed_s)
        >>> print(f"type(ed_s)={type(ed_s)}\\ntype(pd_s)={type(pd_s)}")
        type(ed_s)=<class 'eland.series.Series'>
        type(pd_s)=<class 'pandas.core.series.Series'>
        >>> ed_s
        0     Kibana Airlines
        1    Logstash Airways
        2    Logstash Airways
        3     Kibana Airlines
        4     Kibana Airlines
        Name: Carrier, dtype: object
        >>> pd_s.to_numpy()
        array(['Kibana Airlines', 'Logstash Airways', 'Logstash Airways',
               'Kibana Airlines', 'Kibana Airlines'], dtype=object)
        """
        raise NotImplementedError(
            "This method would scan/scroll the entire Elasticsearch index(s) into memory."
            "If this is explicitly required and there is sufficient memory, call `ed.eland_to_pandas(ed_df).values`"
        )
