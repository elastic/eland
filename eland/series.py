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
import sys

import pandas as pd
import pandas.compat as compat
from pandas.compat import StringIO
from pandas.core.dtypes.common import (
    is_categorical_dtype)
from pandas.io.formats import format as fmt
from pandas.io.formats.printing import pprint_thing

from eland import Index
from eland import NDFrame


class Series(NDFrame):
    """
    pandas.Series like API that proxies into Elasticsearch index(es).

    Parameters
    ----------
    client : eland.Client
        A reference to a Elasticsearch python client

    index_pattern : str
        An Elasticsearch index pattern. This can contain wildcards (e.g. filebeat-*).

    field_name : str
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
                 client,
                 index_pattern,
                 field_name,
                 mappings=None,
                 index_field=None):
        # python 3 syntax
        super().__init__(client, index_pattern, mappings=mappings, index_field=index_field)

        # now select column (field_name)
        self._mappings = self._filter_mappings([field_name])

    def head(self, n=5):
        return self._df_to_series(super()._head(n))

    def tail(self, n=5):
        return self._df_to_series(super()._tail(n))

    def info(self, verbose=None, buf=None, max_cols=None, memory_usage=None,
             null_counts=None):
        """
        Print a concise summary of a DataFrame.

        This method prints information about a DataFrame including
        the index dtype and column dtypes, non-null values and memory usage.

        This copies a lot of code from pandas.DataFrame.info as it is difficult
        to split out the appropriate code or creating a SparseDataFrame gives
        incorrect results on types and counts.
        """
        if buf is None:  # pragma: no cover
            buf = sys.stdout

        lines = []

        lines.append(str(type(self)))
        lines.append(self._index_summary())

        if len(self.columns) == 0:
            lines.append('Empty {name}'.format(name=type(self).__name__))
            fmt.buffer_put_lines(buf, lines)
            return

        cols = self.columns

        # hack
        if max_cols is None:
            max_cols = pd.get_option('display.max_info_columns',
                                     len(self.columns) + 1)

        max_rows = pd.get_option('display.max_info_rows', len(self) + 1)

        if null_counts is None:
            show_counts = ((len(self.columns) <= max_cols) and
                           (len(self) < max_rows))
        else:
            show_counts = null_counts
        exceeds_info_cols = len(self.columns) > max_cols

        def _verbose_repr():
            lines.append('Data columns (total %d columns):' %
                         len(self.columns))
            space = max(len(pprint_thing(k)) for k in self.columns) + 4
            counts = None

            tmpl = "{count}{dtype}"
            if show_counts:
                counts = self.count()
                if len(cols) != len(counts):  # pragma: no cover
                    raise AssertionError(
                        'Columns must equal counts '
                        '({cols:d} != {counts:d})'.format(
                            cols=len(cols), counts=len(counts)))
                tmpl = "{count} non-null {dtype}"

            dtypes = self.dtypes
            for i, col in enumerate(self._columns):
                dtype = dtypes.iloc[i]
                col = pprint_thing(col)

                count = ""
                if show_counts:
                    count = counts.iloc[i]

                lines.append(_put_str(col, space) + tmpl.format(count=count,
                                                                dtype=dtype))

        def _non_verbose_repr():
            lines.append(self._columns._summary(name='Columns'))

        def _sizeof_fmt(num, size_qualifier):
            # returns size in human readable format
            for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
                if num < 1024.0:
                    return ("{num:3.1f}{size_q} "
                            "{x}".format(num=num, size_q=size_qualifier, x=x))
                num /= 1024.0
            return "{num:3.1f}{size_q} {pb}".format(num=num,
                                                    size_q=size_qualifier,
                                                    pb='PB')

        if verbose:
            _verbose_repr()
        elif verbose is False:  # specifically set to False, not nesc None
            _non_verbose_repr()
        else:
            if exceeds_info_cols:
                _non_verbose_repr()
            else:
                _verbose_repr()

        counts = self.get_dtype_counts()
        dtypes = ['{k}({kk:d})'.format(k=k[0], kk=k[1]) for k
                  in sorted(counts.items())]
        lines.append('dtypes: {types}'.format(types=', '.join(dtypes)))

        if memory_usage is None:
            memory_usage = pd.get_option('display.memory_usage')
        if memory_usage:
            # append memory usage of df to display
            size_qualifier = ''

            # TODO - this is different from pd.DataFrame as we shouldn't
            #   really hold much in memory. For now just approximate with getsizeof + ignore deep
            mem_usage = sys.getsizeof(self)
            lines.append("memory usage: {mem}\n".format(
                mem=_sizeof_fmt(mem_usage, size_qualifier)))

        fmt.buffer_put_lines(buf, lines)

    @property
    def shape(self):
        """
        Return a tuple representing the dimensionality of the DataFrame.

        Returns
        -------
        shape: tuple
            0 - number of rows
            1 - number of columns
        """
        num_rows = len(self)
        num_columns = len(self._columns)

        return num_rows, num_columns

    @property
    def set_index(self, index_field):
        copy = self.copy()
        copy._index = Index(index_field)
        return copy

    def _index_summary(self):
        head = self.head(1).index[0]
        tail = self.tail(1).index[0]
        index_summary = ', %s to %s' % (pprint_thing(head),
                                        pprint_thing(tail))

        name = "Index"
        return '%s: %s entries%s' % (name, len(self), index_summary)

    def count(self):
        """
        Count non-NA cells for each column (TODO row)

        Counts are based on exists queries against ES

        This is inefficient, as it creates N queries (N is number of fields).

        An alternative approach is to use value_count aggregations. However, they have issues in that:
        1. They can only be used with aggregatable fields (e.g. keyword not text)
        2. For list fields they return multiple counts. E.g. tags=['elastic', 'ml'] returns value_count=2
        for a single document.
        """
        counts = {}
        for field in self._mappings.source_fields():
            exists_query = {"query": {"exists": {"field": field}}}
            field_exists_count = self._client.count(index=self._index_pattern, body=exists_query)
            counts[field] = field_exists_count

        count = pd.Series(data=counts, index=self._mappings.source_fields())

        return count

    def describe(self):
        return super()._describe()

    def _df_to_series(self, df):
        return df.iloc[:, 0]

    # ----------------------------------------------------------------------
    # Rendering Methods
    def __repr__(self):
        """
        From pandas
        """
        buf = StringIO()

        max_rows = pd.get_option("display.max_rows")

        self.to_string(buf=buf, na_rep='NaN', float_format=None, header=True, index=True, length=False,
                       dtype=False, name=False, max_rows=max_rows)

        return buf.getvalue()

    def to_string(self, buf=None, na_rep='NaN',
                  float_format=None, header=True,
                  index=True, length=True, dtype=True,
                  name=True, max_rows=None):
        """
        From pandas

        Render a string representation of the Series.

        Parameters
        ----------
        buf : StringIO-like, optional
            buffer to write to
        na_rep : string, optional
            string representation of NAN to use, default 'NaN'
        float_format : one-parameter function, optional
            formatter function to apply to columns' elements if they are floats
            default None
        header : boolean, default True
            Add the Series header (index name)
        index : bool, optional
            Add index (row) labels, default True
        length : boolean, default False
            Add the Series length
        dtype : boolean, default False
            Add the Series dtype
        name : boolean, default False
            Add the Series name if not None
        max_rows : int, optional
            Maximum number of rows to show before truncating. If None, show
            all.

        Returns
        -------
        formatted : string (if not buffer passed)
        """
        if max_rows == None:
            max_rows = pd.get_option("display.max_rows")

        df = self._fake_head_tail_df(max_rows=max_rows + 1)

        s = self._df_to_series(df)

        formatter = Series.SeriesFormatter(s, len(self), name=name, length=length,
                                           header=header, index=index,
                                           dtype=dtype, na_rep=na_rep,
                                           float_format=float_format,
                                           max_rows=max_rows)
        result = formatter.to_string()

        # catch contract violations
        if not isinstance(result, compat.text_type):
            raise AssertionError("result must be of type unicode, type"
                                 " of result is {0!r}"
                                 "".format(result.__class__.__name__))

        if buf is None:
            return result
        else:
            try:
                buf.write(result)
            except AttributeError:
                with open(buf, 'w') as f:
                    f.write(result)

    class SeriesFormatter(fmt.SeriesFormatter):
        """
        A hacked overridden version of pandas.io.formats.SeriesFormatter that writes correct length
        """

        def __init__(self, series, series_length, buf=None, length=True, header=True, index=True,
                     na_rep='NaN', name=False, float_format=None, dtype=True,
                     max_rows=None):
            super().__init__(series, buf=buf, length=length, header=header, index=index,
                             na_rep=na_rep, name=name, float_format=float_format, dtype=dtype,
                             max_rows=max_rows)
            self._series_length = series_length

        def _get_footer(self):
            """
            Overridden with length change
            (from pandas 0.24.2 io.formats.SeriesFormatter)
            """
            name = self.series.name
            footer = ''

            if getattr(self.series.index, 'freq', None) is not None:
                footer += 'Freq: {freq}'.format(freq=self.series.index.freqstr)

            if self.name is not False and name is not None:
                if footer:
                    footer += ', '

                series_name = pprint_thing(name,
                                           escape_chars=('\t', '\r', '\n'))
                footer += ("Name: {sname}".format(sname=series_name)
                           if name is not None else "")

            if (self.length is True or
                    (self.length == 'truncate' and self.truncate_v)):
                if footer:
                    footer += ', '
                footer += 'Length: {length}'.format(length=self._series_length)

            if self.dtype is not False and self.dtype is not None:
                name = getattr(self.tr_series.dtype, 'name', None)
                if name:
                    if footer:
                        footer += ', '
                    footer += 'dtype: {typ}'.format(typ=pprint_thing(name))

            # level infos are added to the end and in a new line, like it is done
            # for Categoricals
            if is_categorical_dtype(self.tr_series.dtype):
                level_info = self.tr_series._values._repr_categories_info()
                if footer:
                    footer += "\n"
                footer += level_info

            return compat.text_type(footer)
