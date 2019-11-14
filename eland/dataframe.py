import sys
import warnings
from io import StringIO

import numpy as np
import pandas as pd
import six
from pandas.core.common import apply_if_callable, is_bool_indexer
from pandas.core.dtypes.common import is_list_like
from pandas.core.indexing import check_bool_indexer
from pandas.io.common import _expand_user, _stringify_path
from pandas.io.formats import console
from pandas.io.formats import format as fmt
from pandas.io.formats.printing import pprint_thing

import eland.plotting as gfx
from eland import NDFrame
from eland import Series
from eland.filter import BooleanFilter, ScriptFilter


class DataFrame(NDFrame):
    """
    Two-dimensional size-mutable, potentially heterogeneous tabular data structure with labeled axes
    (rows and columns) referencing data stored in Elasticsearch indices.
    Where possible APIs mirror pandas.DataFrame APIs.
    The underlying data is stored in Elasticsearch rather than core memory.

    Parameters
    ----------
    client: Elasticsearch client argument(s) (e.g. 'localhost:9200')
        - elasticsearch-py parameters or
        - elasticsearch-py instance or
        - eland.Client instance
    index_pattern: str
        Elasticsearch index pattern (e.g. 'flights' or 'filebeat-\*')
    columns: list of str, optional
        List of DataFrame columns. A subset of the Elasticsearch index's fields.
    index_field: str, optional
        The Elasticsearch index field to use as the DataFrame index. Defaults to _id if None is used.

    See Also
    --------
    :pandas_docs:`pandas.DataFrame`

    Examples
    --------
    Constructing DataFrame from an Elasticsearch configuration arguments and an Elasticsearch index

    >>> df = ed.DataFrame('localhost:9200', 'flights')
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
    >>> es = Elasticsearch("localhost:9200")
    >>> df = ed.DataFrame(client=es, index_pattern='flights', columns=['AvgTicketPrice', 'Cancelled'])
    >>> df.head()
       AvgTicketPrice  Cancelled
    0      841.265642      False
    1      882.982662      False
    2      190.636904      False
    3      181.694216       True
    4      730.041778      False
    <BLANKLINE>
    [5 rows x 2 columns]

    Constructing DataFrame from an Elasticsearch client and an Elasticsearch index, with 'timestamp' as the  DataFrame index field

    >>> df = ed.DataFrame(client='localhost', index_pattern='flights', columns=['AvgTicketPrice', 'timestamp'], index_field='timestamp')
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

    def __init__(self,
                 client=None,
                 index_pattern=None,
                 columns=None,
                 index_field=None,
                 query_compiler=None):
        """
        There are effectively 2 constructors:

        1. client, index_pattern, columns, index_field
        2. query_compiler (eland.ElandQueryCompiler)

        The constructor with 'query_compiler' is for internal use only.
        """
        if query_compiler is None:
            if client is None or index_pattern is None:
                raise ValueError("client and index_pattern must be defined in DataFrame constructor")
        # python 3 syntax
        super().__init__(
            client=client,
            index_pattern=index_pattern,
            columns=columns,
            index_field=index_field,
            query_compiler=query_compiler)

    def _get_columns(self):
        """
        The column labels of the DataFrame.

        Returns
        -------
        Elasticsearch field names as pandas.Index

        See Also
        --------
        :pandas_docs:`pandas.DataFrame.columns`

        Examples
        --------
        >>> df = ed.DataFrame('localhost', 'flights')
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

    columns = property(_get_columns)

    @property
    def empty(self):
        """Determines if the DataFrame is empty.

        Returns
        -------
        bool
            If DataFrame is empty, return True, if not return False.

        See Also
        --------
        :pandas_docs:`pandas.DataFrame.empty`

        Examples
        --------
        >>> df = ed.DataFrame('localhost', 'flights')
        >>> df.empty
        False
        """
        return len(self.columns) == 0 or len(self.index) == 0

    def head(self, n=5):
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
        :pandas_docs:`pandas.DataFrame.head`

        Examples
        --------
        >>> df = ed.DataFrame('localhost', 'flights', columns=['Origin', 'Dest'])
        >>> df.head(3)
                                    Origin                                          Dest
        0        Frankfurt am Main Airport  Sydney Kingsford Smith International Airport
        1  Cape Town International Airport                     Venice Marco Polo Airport
        2        Venice Marco Polo Airport                     Venice Marco Polo Airport
        <BLANKLINE>
        [3 rows x 2 columns]
        """
        return DataFrame(query_compiler=self._query_compiler.head(n))

    def tail(self, n=5):
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
        :pandas_docs:`pandas.DataFrame.tail`

        Examples
        --------
        >>> df = ed.DataFrame('localhost', 'flights', columns=['Origin', 'Dest'])
        >>> df.tail()
                                                          Origin                                      Dest
        13054                         Pisa International Airport      Xi'an Xianyang International Airport
        13055  Winnipeg / James Armstrong Richardson Internat...                            Zurich Airport
        13056     Licenciado Benito Juarez International Airport                         Ukrainka Air Base
        13057                                      Itami Airport  Ministro Pistarini International Airport
        13058                     Adelaide International Airport   Washington Dulles International Airport
        <BLANKLINE>
        [5 rows x 2 columns]
        """
        return DataFrame(query_compiler=self._query_compiler.tail(n))

    def __repr__(self):
        """
        From pandas
        """
        buf = StringIO()

        max_rows = pd.get_option("display.max_rows")
        max_cols = pd.get_option("display.max_columns")
        show_dimensions = pd.get_option("display.show_dimensions")
        if pd.get_option("display.expand_frame_repr"):
            width, _ = console.get_console_size()
        else:
            width = None
        self.to_string(buf=buf, max_rows=max_rows, max_cols=max_cols,
                       line_width=width, show_dimensions=show_dimensions)

        return buf.getvalue()

    def _info_repr(self):
        """
        True if the repr should show the info view.
        """
        info_repr_option = (pd.get_option("display.large_repr") == "info")
        return info_repr_option and not (self._repr_fits_horizontal_() and
                                         self._repr_fits_vertical_())

    def _repr_html_(self):
        """
        From pandas
        """
        if self._info_repr():
            buf = StringIO("")
            self.info(buf=buf)
            # need to escape the <class>, should be the first line.
            val = buf.getvalue().replace('<', r'&lt;', 1)
            val = val.replace('>', r'&gt;', 1)
            return '<pre>' + val + '</pre>'

        if pd.get_option("display.notebook_repr_html"):
            max_rows = pd.get_option("display.max_rows")
            max_cols = pd.get_option("display.max_columns")
            show_dimensions = pd.get_option("display.show_dimensions")

            return self.to_html(max_rows=max_rows, max_cols=max_cols,
                                show_dimensions=show_dimensions, notebook=True)
        else:
            return None

    def count(self):
        """
        Count non-NA cells for each column.

        Counts are based on exists queries against ES.

        This is inefficient, as it creates N queries (N is number of fields).
        An alternative approach is to use value_count aggregations. However, they have issues in that:

        - They can only be used with aggregatable fields (e.g. keyword not text)
        - For list fields they return multiple counts. E.g. tags=['elastic', 'ml'] returns value_count=2 for a single document.

        TODO - add additional pandas.DataFrame.count features

        Returns
        -------
        pandas.Series:
            Summary of column counts

        See Also
        --------
        :pandas_docs:`pandas.DataFrame.count`

        Examples
        --------
        >>> df = ed.DataFrame('localhost', 'ecommerce', columns=['customer_first_name', 'geoip.city_name'])
        >>> df.count()
        customer_first_name    4675
        geoip.city_name        4094
        dtype: int64
        """
        return self._query_compiler.count()

    def info_es(self):
        """

        Returns
        -------
        None
            This method prints a debug summary of the task list Elasticsearch
        """
        buf = StringIO()

        super()._info_es(buf)

        return buf.getvalue()

    def _index_summary(self):
        # Print index summary e.g.
        # Index: 103 entries, 0 to 102
        # Do this by getting head and tail of dataframe
        head = self.head(1)._to_pandas().index[0]
        tail = self.tail(1)._to_pandas().index[0]
        index_summary = ', %s to %s' % (pprint_thing(head),
                                        pprint_thing(tail))

        name = "Index"
        return '%s: %s entries%s' % (name, len(self), index_summary)

    def info(self, verbose=None, buf=None, max_cols=None, memory_usage=None,
             null_counts=None):
        """
        Print a concise summary of a DataFrame.

        This method prints information about a DataFrame including
        the index dtype and column dtypes, non-null values and memory usage.

        See :pandas_docs:`pandas.DataFrame.info` for full details.

        Notes
        -----
        This copies a lot of code from pandas.DataFrame.info as it is difficult
        to split out the appropriate code or creating a SparseDataFrame gives
        incorrect results on types and counts.

        Examples
        --------
        >>> df = ed.DataFrame('localhost', 'ecommerce', columns=['customer_first_name', 'geoip.city_name'])
        >>> df.info()
        <class 'eland.dataframe.DataFrame'>
        Index: 4675 entries, 0 to 4674
        Data columns (total 2 columns):
        customer_first_name    4675 non-null object
        geoip.city_name        4094 non-null object
        dtypes: object(2)
        memory usage: 96.0 bytes
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

        # From pandas.DataFrame
        def _put_str(s, space):
            return '{s}'.format(s=s)[:space].ljust(space)

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
            for i, col in enumerate(self.columns):
                dtype = dtypes.iloc[i]
                col = pprint_thing(col)

                count = ""
                if show_counts:
                    count = counts.iloc[i]

                lines.append(_put_str(col, space) + tmpl.format(count=count,
                                                                dtype=dtype))

        def _non_verbose_repr():
            lines.append(self.columns._summary(name='Columns'))

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

        counts = self.dtypes.value_counts()
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

    def to_html(self, buf=None, columns=None, col_space=None, header=True,
                index=True, na_rep='NaN', formatters=None, float_format=None,
                sparsify=None, index_names=True, justify=None, max_rows=None,
                max_cols=None, show_dimensions=False, decimal='.',
                bold_rows=True, classes=None, escape=True, notebook=False,
                border=None, table_id=None, render_links=False):
        """
        From pandas - except we set max_rows default to avoid careless extraction of entire index
        """
        if max_rows is None:
            warnings.warn("DataFrame.to_string called without max_rows set "
                          "- this will return entire index results. "
                          "Setting max_rows=60, overwrite if different behaviour is required.")
            max_rows = 60

        # Create a slightly bigger dataframe than display
        df = self._build_repr_df(max_rows + 1, max_cols)

        if buf is not None:
            _buf = _expand_user(_stringify_path(buf))
        else:
            _buf = StringIO()

        df.to_html(buf=_buf, columns=columns, col_space=col_space, header=header,
                   index=index, na_rep=na_rep, formatters=formatters, float_format=float_format,
                   sparsify=sparsify, index_names=index_names, justify=justify, max_rows=max_rows,
                   max_cols=max_cols, show_dimensions=False, decimal=decimal,
                   bold_rows=bold_rows, classes=classes, escape=escape, notebook=notebook,
                   border=border, table_id=table_id, render_links=render_links)

        # Our fake dataframe has incorrect number of rows (max_rows*2+1) - write out
        # the correct number of rows
        if show_dimensions:
            _buf.write("\n<p>{nrows} rows x {ncols} columns</p>"
                       .format(nrows=len(self.index), ncols=len(self.columns)))

        if buf is None:
            result = _buf.getvalue()
            return result

    def to_string(self, buf=None, columns=None, col_space=None, header=True,
                  index=True, na_rep='NaN', formatters=None, float_format=None,
                  sparsify=None, index_names=True, justify=None,
                  max_rows=None, max_cols=None, show_dimensions=False,
                  decimal='.', line_width=None):
        """
        From pandas - except we set max_rows default to avoid careless extraction of entire index
        """
        if max_rows is None:
            warnings.warn("DataFrame.to_string called without max_rows set "
                          "- this will return entire index results. "
                          "Setting max_rows=60, overwrite if different behaviour is required.")
            max_rows = 60

        # Create a slightly bigger dataframe than display
        df = self._build_repr_df(max_rows + 1, max_cols)

        if buf is not None:
            _buf = _expand_user(_stringify_path(buf))
        else:
            _buf = StringIO()

        df.to_string(buf=_buf, columns=columns,
                     col_space=col_space, na_rep=na_rep,
                     formatters=formatters,
                     float_format=float_format,
                     sparsify=sparsify, justify=justify,
                     index_names=index_names,
                     header=header, index=index,
                     max_rows=max_rows,
                     max_cols=max_cols,
                     show_dimensions=False,  # print this outside of this call
                     decimal=decimal,
                     line_width=line_width)

        # Our fake dataframe has incorrect number of rows (max_rows*2+1) - write out
        # the correct number of rows
        if show_dimensions:
            _buf.write("\n\n[{nrows} rows x {ncols} columns]"
                       .format(nrows=len(self.index), ncols=len(self.columns)))

        if buf is None:
            result = _buf.getvalue()
            return result

    def _getitem(self, key):
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
            return DataFrame(
                query_compiler=self._query_compiler._update_query(key)
            )
        else:
            return self._getitem_column(key)

    def _getitem_column(self, key):
        if key not in self.columns:
            raise KeyError("Requested column is not in the DataFrame {}".format(key))
        s = self._reduce_dimension(self._query_compiler.getitem_column_array([key]))
        return s

    def _getitem_array(self, key):
        if isinstance(key, Series):
            key = key._to_pandas()
        if is_bool_indexer(key):
            if isinstance(key, pd.Series) and not key.index.equals(self.index):
                warnings.warn(
                    "Boolean Series key will be reindexed to match DataFrame index.",
                    PendingDeprecationWarning,
                    stacklevel=3,
                )
            elif len(key) != len(self.index):
                raise ValueError(
                    "Item wrong length {} instead of {}.".format(
                        len(key), len(self.index)
                    )
                )
            key = check_bool_indexer(self.index, key)
            # We convert to a RangeIndex because getitem_row_array is expecting a list
            # of indices, and RangeIndex will give us the exact indices of each boolean
            # requested.
            key = pd.RangeIndex(len(self.index))[key]
            if len(key):
                return DataFrame(
                    query_compiler=self._query_compiler.getitem_row_array(key)
                )
            else:
                return DataFrame(columns=self.columns)
        else:
            if any(k not in self.columns for k in key):
                raise KeyError(
                    "{} not index".format(
                        str([k for k in key if k not in self.columns]).replace(",", "")
                    )
                )
            return DataFrame(
                query_compiler=self._query_compiler.getitem_column_array(key)
            )

    def _create_or_update_from_compiler(self, new_query_compiler, inplace=False):
        """Returns or updates a DataFrame given new query_compiler"""
        assert (
                isinstance(new_query_compiler, type(self._query_compiler))
                or type(new_query_compiler) in self._query_compiler.__class__.__bases__
        ), "Invalid Query Compiler object: {}".format(type(new_query_compiler))
        if not inplace:
            return DataFrame(query_compiler=new_query_compiler)
        else:
            self._query_compiler = new_query_compiler

    def _reduce_dimension(self, query_compiler):
        return Series(query_compiler=query_compiler)

    def to_csv(self, path_or_buf=None, sep=",", na_rep='', float_format=None,
               columns=None, header=True, index=True, index_label=None,
               mode='w', encoding=None, compression='infer', quoting=None,
               quotechar='"', line_terminator=None, chunksize=None,
               tupleize_cols=None, date_format=None, doublequote=True,
               escapechar=None, decimal='.'):
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

    def _to_pandas(self):
        """
        Utility method to convert eland.Dataframe to pandas.Dataframe

        Returns
        -------
        pandas.DataFrame
        """
        return self._query_compiler.to_pandas()

    def _empty_pd_df(self):
        return self._query_compiler._empty_pd_ef()

    def squeeze(self, axis=None):
        return DataFrame(
            query_compiler=self._query_compiler.squeeze(axis)
        )

    def select_dtypes(self, include=None, exclude=None):
        """
        Return a subset of the DataFrame's columns based on the column dtypes.

        Compatible with :pandas_docs:`pandas.DataFrame.select_dtypes`
        """
        empty_df = self._empty_pd_df()

        empty_df = empty_df.select_dtypes(include=include, exclude=exclude)

        return self._getitem_array(empty_df.columns)

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
        num_columns = len(self.columns)

        return num_rows, num_columns

    def keys(self):
        return self.columns

    def aggregate(self, func, axis=0, *args, **kwargs):
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
        axis
            Currently, we only support axis=0 (index)
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
        :pandas_docs:`pandas.DataFrame.aggregate`

        Examples
        --------
        >>> df = ed.DataFrame('localhost', 'flights')
        >>> df[['DistanceKilometers', 'AvgTicketPrice']].aggregate(['sum', 'min', 'std'])
             DistanceKilometers  AvgTicketPrice
        sum        9.261629e+07    8.204365e+06
        min        0.000000e+00    1.000205e+02
        std        4.578263e+03    2.663867e+02
        """
        axis = pd.DataFrame._get_axis_number(axis)

        if axis == 1:
            raise NotImplementedError("Aggregating via index not currently implemented - needs index transform")

        # currently we only support a subset of functions that aggregate columns.
        # ['count', 'mad', 'max', 'mean', 'median', 'min', 'mode', 'quantile', 'rank', 'sem', 'skew', 'sum', 'std', 'var', 'nunique']
        if isinstance(func, str):
            # wrap in list
            func = [func]
            return self._query_compiler.aggs(func)
        elif is_list_like(func):
            # we have a list!
            return self._query_compiler.aggs(func)

    agg = aggregate

    hist = gfx.ed_hist_frame

    def query(self, expr, inplace=False, **kwargs):
        """Queries the Dataframe with a boolean expression

        Returns:
            A new DataFrame if inplace=False
        """
        if isinstance(expr, BooleanFilter):
            return DataFrame(
                query_compiler=self._query_compiler._update_query(BooleanFilter(expr))
            )
        elif isinstance(expr, six.string_types):
            return DataFrame(
                query_compiler=self._query_compiler._update_query(ScriptFilter(expr))
            )
        else:
            raise NotImplementedError(expr, type(expr))

    def get(self, key, default=None):
        """
        Get item from object for given key (ex: DataFrame column).
        Returns default value if not found.

        Parameters
        ----------
        key: object

        Returns
        -------
        value: same type as items contained in object

        See Also
        --------
        :pandas_docs:`pandas.DataFrame.get`

        Examples
        --------
        >>> df = ed.DataFrame('localhost', 'flights')
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
