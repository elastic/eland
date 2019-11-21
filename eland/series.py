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
Based on NDFrame which underpins eland.1DataFrame

"""

from io import StringIO

import pandas as pd

from eland import NDFrame
from eland.filter import NotFilter, Equal, Greater, Less, GreaterEqual, LessEqual, ScriptFilter, IsIn


class Series(NDFrame):
    """
    pandas.Series like API that proxies into Elasticsearch index(es).

    Parameters
    ----------
    client : eland.Client
        A reference to a Elasticsearch python client

    index_pattern : str
        An Elasticsearch index pattern. This can contain wildcards (e.g. filebeat-*).

    index_field : str
        The field to base the series on

    See Also
    --------

    Examples
    --------

    import eland as ed
    client = ed.Client(Elasticsearch())
    s = ed.DataFrame(client, 'reviews', 'date')
    df.head()
       reviewerId  vendorId  rating              date
    0           0         0       5  2006-04-07 17:08
    1           1         1       5  2006-05-04 12:16
    2           2         2       4  2006-04-21 12:26
    3           3         3       5  2006-04-18 15:48
    4           3         4       5  2006-04-18 15:49

    Notice that the types are based on Elasticsearch mappings

    Notes
    -----
    If the Elasticsearch index is deleted or index mappings are changed after this
    object is created, the object is not rebuilt and so inconsistencies can occur.

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

    @property
    def empty(self):
        """Determines if the Series is empty.

        Returns:
            True if the Series is empty.
            False otherwise.
        """
        return len(self.index) == 0

    def _get_name(self):
        return self._query_compiler.columns[0]

    name = property(_get_name)

    def rename(self, new_name):
        """
        ONLY COLUMN rename supported

        Parameters
        ----------
        new_name

        Returns
        -------

        """
        return Series(query_compiler=self._query_compiler.rename({self.name: new_name}))

    def head(self, n=5):
        return Series(query_compiler=self._query_compiler.head(n))

    def tail(self, n=5):
        return Series(query_compiler=self._query_compiler.tail(n))

    def value_counts(self, es_size=10):
        """
        Return the value counts for the specified field.

        **Note we can only do this for aggregatable Elasticsearch fields - (in general) numeric and keyword rather than text fields**

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
            number of occurences of each value in the column

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

    # ----------------------------------------------------------------------
    # Rendering Methods
    def __repr__(self):
        num_rows = pd.get_option("max_rows") or 60

        return self.to_string(max_rows=num_rows)

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
            max_rows=None):

        if max_rows is None:
            warnings.warn("Series.to_string called without max_rows set "
                          "- this will return entire index results. "
                          "Setting max_rows=60, overwrite if different behaviour is required.")
            max_rows = 60

        # Create a slightly bigger dataframe than display
        temp_df = self._build_repr_df(max_rows + 1, None)
        if isinstance(temp_df, pd.DataFrame):
            temp_df = temp_df[self.name]
        temp_str = repr(temp_df)
        if self.name is not None:
            name_str = "Name: {}, ".format(str(self.name))
        else:
            name_str = ""
        if len(self.index) > max_rows:
            len_str = "Length: {}, ".format(len(self.index))
        else:
            len_str = ""
        dtype_str = "dtype: {}".format(temp_str.rsplit("dtype: ", 1)[-1])
        if len(self) == 0:
            return "Series([], {}{}".format(name_str, dtype_str)
        return temp_str.rsplit("\nName:", 1)[0] + "\n{}{}{}".format(
            name_str, len_str, dtype_str
        )

    def _to_pandas(self):
        return self._query_compiler.to_pandas()[self.name]

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
        Returns 1 by definition of a Series1

        Returns
        -------
        int
            By definition 1

        """
        return 1

    def info_es(self):
        buf = StringIO()

        super()._info_es(buf)

        return buf.getvalue()

    def __truediv__(self, right):
        return self.truediv(right)

    def truediv(self, right):
        """
        return a / b

        a & b == Series
            a & b must share same eland.Client, index_pattern and index_field
        """
        if isinstance(right, Series):
            # Check compatibility
            self._query_compiler.check_arithmetics(right._query_compiler)

            field_name = "{0}_{1}_{2}".format(self.name, "truediv", right.name)

            # Compatible, so create new Series
            return Series(query_compiler=self._query_compiler.arithmetic_op_fields(
                field_name, 'truediv', self.name, right.name))
        else:
            raise TypeError(
                "Can only perform arithmetic operation on selected types "
                "{0} != {1}".format(type(self), type(right))
            )
