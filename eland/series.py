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

import sys
import warnings
from io import StringIO

import numpy as np
import pandas as pd
from pandas.io.common import _expand_user, _stringify_path

import eland.plotting
from eland import NDFrame
from eland.arithmetics import ArithmeticSeries, ArithmeticString, ArithmeticNumber
from eland.common import DEFAULT_NUM_ROWS_DISPLAYED, docstring_parameter
from eland.filter import NotFilter, Equal, Greater, Less, GreaterEqual, LessEqual, ScriptFilter, IsIn


def _get_method_name():
    return sys._getframe(1).f_code.co_name


class Series(NDFrame):
    """
    pandas.Series like API that proxies into Elasticsearch index(es).

    Parameters
    ----------
    client : eland.Client
        A reference to a Elasticsearch python client

    index_pattern : str
        An Elasticsearch index pattern. This can contain wildcards.

    index_field : str
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
    >>> ed.Series(client='localhost', index_pattern='flights', name='Carrier')
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

    def __init__(self,
                 client=None,
                 index_pattern=None,
                 name=None,
                 index_field=None,
                 query_compiler=None):
        # Series has 1 column
        if name is None:
            columns = None
        else:
            columns = [name]

        super().__init__(
            client=client,
            index_pattern=index_pattern,
            columns=columns,
            index_field=index_field,
            query_compiler=query_compiler)

    hist = eland.plotting.ed_hist_series

    @property
    def empty(self):
        """Determines if the Series is empty.

        Returns:
            True if the Series is empty.
            False otherwise.
        """
        return len(self.index) == 0

    @property
    def shape(self):
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
    def field_name(self):
        """
        Returns
        -------
        field_name: str
            Return the Elasticsearch field name for this series
        """
        return self._query_compiler.field_names[0]

    def _get_name(self):
        return self._query_compiler.columns[0]

    def _set_name(self, name):
        self._query_compiler.rename({self.name: name}, inplace=True)

    name = property(_get_name, _set_name)

    def rename(self, new_name):
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
        return Series(query_compiler=self._query_compiler.rename({self.name: new_name}))

    def head(self, n=5):
        return Series(query_compiler=self._query_compiler.head(n))

    def tail(self, n=5):
        return Series(query_compiler=self._query_compiler.tail(n))

    def value_counts(self, es_size=10):
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
        if not es_size > 0:
            raise ValueError("es_size must be a positive integer.")

        return self._query_compiler.value_counts(es_size)

    # dtype not implemented for Series as causes query to fail
    # in pandas.core.computation.ops.Term.type

    # ----------------------------------------------------------------------
    # Rendering Methods
    def __repr__(self):
        """
        Return a string representation for a particular Series.
        """
        buf = StringIO()

        # max_rows and max_cols determine the maximum size of the pretty printed tabular
        # representation of the series. pandas defaults are 60 and 20 respectively.
        # series where len(series) > max_rows shows a truncated view with 10 rows shown.
        max_rows = pd.get_option("display.max_rows")
        min_rows = pd.get_option("display.min_rows")

        if len(self) > max_rows:
            max_rows = min_rows

        show_dimensions = pd.get_option("display.show_dimensions")

        self.to_string(
            buf=buf,
            name=self.name,
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
    ):
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
            warnings.warn("Series.to_string called without max_rows set "
                          "- this will return entire index results. "
                          "Setting max_rows={default}"
                          " overwrite if different behaviour is required."
                          .format(default=DEFAULT_NUM_ROWS_DISPLAYED),
                          UserWarning)
            max_rows = DEFAULT_NUM_ROWS_DISPLAYED

        # because of the way pandas handles max_rows=0, not having this throws an error
        # see eland issue #56
        if max_rows == 0:
            max_rows = 1

        # Create a slightly bigger dataframe than display
        temp_series = self._build_repr(max_rows + 1)

        if buf is not None:
            _buf = _expand_user(_stringify_path(buf))
        else:
            _buf = StringIO()

        # Create repr of fake series without name, length, dtype summary
        temp_str = temp_series.to_string(buf=_buf,
                                         na_rep=na_rep,
                                         float_format=float_format,
                                         header=header,
                                         index=index,
                                         length=False,
                                         dtype=False,
                                         name=False,
                                         max_rows=max_rows)

        # Create the summary
        footer = ""
        if name and self.name is not None:
            footer += "Name: {}".format(str(self.name))
        if length and len(self) > max_rows:
            if footer:
                footer += ", "
            footer += "Length: {}".format(len(self.index))
        if dtype:
            if footer:
                footer += ", "
            footer += "dtype: {}".format(temp_series.dtype)

        if len(footer) > 0:
            _buf.write("\n{}".format(footer))

        if buf is None:
            result = _buf.getvalue()
            return result

    def _to_pandas(self, show_progress=False):
        return self._query_compiler.to_pandas(show_progress=show_progress)[self.name]

    @property
    def _dtype(self):
        # DO NOT MAKE PUBLIC (i.e. def dtype) as this breaks query eval implementation
        return self._query_compiler.dtypes[0]

    def __gt__(self, other):
        if isinstance(other, Series):
            # Need to use scripted query to compare to values
            painless = "doc['{}'].value > doc['{}'].value".format(self.name, other.name)
            return ScriptFilter(painless)
        elif isinstance(other, (int, float)):
            return Greater(field=self.name, value=other)
        else:
            raise NotImplementedError(other, type(other))

    def __lt__(self, other):
        if isinstance(other, Series):
            # Need to use scripted query to compare to values
            painless = "doc['{}'].value < doc['{}'].value".format(self.name, other.name)
            return ScriptFilter(painless)
        elif isinstance(other, (int, float)):
            return Less(field=self.name, value=other)
        else:
            raise NotImplementedError(other, type(other))

    def __ge__(self, other):
        if isinstance(other, Series):
            # Need to use scripted query to compare to values
            painless = "doc['{}'].value >= doc['{}'].value".format(self.name, other.name)
            return ScriptFilter(painless)
        elif isinstance(other, (int, float)):
            return GreaterEqual(field=self.name, value=other)
        else:
            raise NotImplementedError(other, type(other))

    def __le__(self, other):
        if isinstance(other, Series):
            # Need to use scripted query to compare to values
            painless = "doc['{}'].value <= doc['{}'].value".format(self.name, other.name)
            return ScriptFilter(painless)
        elif isinstance(other, (int, float)):
            return LessEqual(field=self.name, value=other)
        else:
            raise NotImplementedError(other, type(other))

    def __eq__(self, other):
        if isinstance(other, Series):
            # Need to use scripted query to compare to values
            painless = "doc['{}'].value == doc['{}'].value".format(self.name, other.name)
            return ScriptFilter(painless)
        elif isinstance(other, (int, float)):
            return Equal(field=self.name, value=other)
        elif isinstance(other, str):
            return Equal(field=self.name, value=other)
        else:
            raise NotImplementedError(other, type(other))

    def __ne__(self, other):
        if isinstance(other, Series):
            # Need to use scripted query to compare to values
            painless = "doc['{}'].value != doc['{}'].value".format(self.name, other.name)
            return ScriptFilter(painless)
        elif isinstance(other, (int, float)):
            return NotFilter(Equal(field=self.name, value=other))
        elif isinstance(other, str):
            return NotFilter(Equal(field=self.name, value=other))
        else:
            raise NotImplementedError(other, type(other))

    def isin(self, other):
        if isinstance(other, list):
            return IsIn(field=self.name, value=other)
        else:
            raise NotImplementedError(other, type(other))

    @property
    def ndim(self):
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

    def info_es(self):
        buf = StringIO()

        super()._info_es(buf)

        return buf.getvalue()

    def __add__(self, right):
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

    def __truediv__(self, right):
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

    def __floordiv__(self, right):
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

    def __mod__(self, right):
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

    def __mul__(self, right):
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

    def __sub__(self, right):
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

    def __pow__(self, right):
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

    def __radd__(self, left):
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

    def __rtruediv__(self, left):
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

    def __rfloordiv__(self, left):
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

    def __rmod__(self, left):
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

    def __rmul__(self, left):
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

    def __rpow__(self, left):
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
        >>> np.int(2) ** df.total_quantity
        0    4.0
        1    4.0
        2    4.0
        3    4.0
        4    4.0
        Name: total_quantity, dtype: float64
        """
        return self._numeric_op(left, _get_method_name())

    def __rsub__(self, left):
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

    def _numeric_op(self, right, method_name):
        """
        return a op b

        a & b == Series
            a & b must share same eland.Client, index_pattern and index_field
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

            right_object = ArithmeticSeries(right._query_compiler, right.name, right._dtype)
            display_name = None
        elif np.issubdtype(np.dtype(type(right)), np.number):
            right_object = ArithmeticNumber(right, np.dtype(type(right)))
            display_name = self.name
        elif isinstance(right, str):
            right_object = ArithmeticString(right)
            display_name = self.name
        else:
            raise TypeError(
                "unsupported operation type(s) ['{}'] for operands ['{}' with dtype '{}', '{}']"
                    .format(method_name, type(self), self._dtype, type(right).__name__)
            )

        left_object = ArithmeticSeries(self._query_compiler, self.name, self._dtype)
        left_object.arithmetic_operation(method_name, right_object)

        series = Series(query_compiler=self._query_compiler.arithmetic_op_fields(display_name, left_object))

        # force set name to 'display_name'
        series._query_compiler._mappings.display_names = [display_name]

        return series

    def max(self, numeric_only=None):
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
        >>> s = ed.Series('localhost', 'flights', name='AvgTicketPrice')
        >>> int(s.max())
        1199
        """
        results = super().max(numeric_only=numeric_only)
        return results.squeeze()

    def mean(self, numeric_only=None):
        """
        Return the mean of the Series values

        TODO - implement remainder of pandas arguments, currently non-numerics are not supported

        Returns
        -------
        float
            max value

        See Also
        --------
        :pandas_api_docs:`pandas.Series.mean`

        Examples
        --------
        >>> s = ed.Series('localhost', 'flights', name='AvgTicketPrice')
        >>> int(s.mean())
        628
        """
        results = super().mean(numeric_only=numeric_only)
        return results.squeeze()

    def min(self, numeric_only=None):
        """
        Return the minimum of the Series values

        TODO - implement remainder of pandas arguments, currently non-numerics are not supported

        Returns
        -------
        float
            max value

        See Also
        --------
        :pandas_api_docs:`pandas.Series.min`

        Examples
        --------
        >>> s = ed.Series('localhost', 'flights', name='AvgTicketPrice')
        >>> int(s.min())
        100
        """
        results = super().min(numeric_only=numeric_only)
        return results.squeeze()

    def sum(self, numeric_only=None):
        """
        Return the sum of the Series values

        TODO - implement remainder of pandas arguments, currently non-numerics are not supported

        Returns
        -------
        float
            max value

        See Also
        --------
        :pandas_api_docs:`pandas.Series.sum`

        Examples
        --------
        >>> s = ed.Series('localhost', 'flights', name='AvgTicketPrice')
        >>> int(s.sum())
        8204364
        """
        results = super().sum(numeric_only=numeric_only)
        return results.squeeze()

    def nunique(self):
        """
        Return the sum of the Series values

        Returns
        -------
        float
            max value

        See Also
        --------
        :pandas_api_docs:`pandas.Series.sum`

        Examples
        --------
        >>> s = ed.Series('localhost', 'flights', name='Carrier')
        >>> s.nunique()
        4
        """
        results = super().nunique()
        return results.squeeze()

    # def values TODO - not implemented as causes current implementation of query to fail

    def to_numpy(self):
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
        >>> print("type(ed_s)={0}\\ntype(pd_s)={1}".format(type(ed_s), type(pd_s)))
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
