"""
DataFrame
---------
An efficient 2D container for potentially mixed-type time series or other
labeled data series.

The underlying data resides in Elasticsearch and the API aligns as much as
possible with pandas.DataFrame API.

This allows the eland.DataFrame to access large datasets stored in Elasticsearch,
without storing the dataset in local memory.

Implementation Details
----------------------

Elasticsearch indexes can be configured in many different ways, and these indexes
utilise different data structures to pandas.DataFrame.

eland.DataFrame operations that return individual rows (e.g. df.head()) return
_source data. If _source is not enabled, this data is not accessible.

Similarly, only Elasticsearch searchable fields can be searched or filtered, and
only Elasticsearch aggregatable fields can be aggregated or grouped.

"""
import sys

import pandas as pd

from pandas.io.formats import format as fmt
from pandas.io.formats.printing import pprint_thing
from pandas.compat import StringIO
from pandas.io.common import _expand_user, _stringify_path
from pandas.io.formats import console
from pandas.core import common as com

from eland import NDFrame
from eland import Index
from eland import Series





class DataFrame(NDFrame):
    """
    pandas.DataFrame like API that proxies into Elasticsearch index(es).

    Parameters
    ----------
    client : eland.Client
        A reference to a Elasticsearch python client

    index_pattern : str
        An Elasticsearch index pattern. This can contain wildcards (e.g. filebeat-*).

    See Also
    --------

    Examples
    --------

    import eland as ed
    client = ed.Client(Elasticsearch())
    df = ed.DataFrame(client, 'reviews')
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
                 mappings=None,
                 index_field=None):
        # python 3 syntax
        super().__init__(client, index_pattern, mappings=mappings, index_field=index_field)

    def head(self, n=5):
        return super()._head(n)

    def tail(self, n=5):
        return super()._tail(n)

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
        num_columns = len(self.columns)

        return num_rows, num_columns

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


    def __getitem__(self, key):
        # NOTE: there is a difference between pandas here.
        # e.g. df['a'] returns pd.Series, df[['a','b']] return pd.DataFrame

        # Implementation mainly copied from pandas v0.24.2
        # (https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html)
        key = com.apply_if_callable(key, self)

        # TODO - add slice capabilities - need to add index features first
        #   e.g. set index etc.
        # Do we have a slicer (on rows)?
        """
        indexer = convert_to_index_sliceable(self, key)
        if indexer is not None:
            return self._slice(indexer, axis=0)
        # Do we have a (boolean) DataFrame?
        if isinstance(key, DataFrame):
            return self._getitem_frame(key)
        """

        # Do we have a (boolean) 1d indexer?
        """
        if com.is_bool_indexer(key):
            return self._getitem_bool_array(key)
        """

        # We are left with two options: a single key, and a collection of keys,
        columns = []
        is_single_key = False
        if isinstance(key, str):
            if not self._mappings.is_source_field(key):
                raise TypeError('Column does not exist: [{0}]'.format(key))
            columns.append(key)
            is_single_key = True
        elif isinstance(key, list):
            columns.extend(key)
        else:
            raise TypeError('__getitem__ arguments invalid: [{0}]'.format(key))

        mappings = self._filter_mappings(columns)

        # Return new eland.DataFrame with modified mappings
        if is_single_key:
            return Series(self._client, self._index_pattern, mappings=mappings)
        else:
            return DataFrame(self._client, self._index_pattern, mappings=mappings)


    def __getattr__(self, name):
        # Note: obj.x will always call obj.__getattribute__('x') prior to
        # calling obj.__getattr__('x').
        mappings = self._filter_mappings([name])

        return Series(self._client, self._index_pattern, mappings=mappings)

    def copy(self):
        # TODO - test and validate...may need deep copying
        return DataFrame(self._client,
                 self._index_pattern,
                 self._mappings,
                 self._index)

    # ----------------------------------------------------------------------
    # Rendering Methods
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

    def to_string(self, buf=None, columns=None, col_space=None, header=True,
                  index=True, na_rep='NaN', formatters=None, float_format=None,
                  sparsify=None, index_names=True, justify=None,
                  max_rows=None, max_cols=None, show_dimensions=True,
                  decimal='.', line_width=None):
        """
        From pandas
        """
        if max_rows == None:
            max_rows = pd.get_option('display.max_rows')

        df = self._fake_head_tail_df(max_rows=max_rows+1)

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
                       .format(nrows=self._index_count(), ncols=len(self.columns)))

        if buf is None:
            result = _buf.getvalue()
            return result

    def to_pandas(selfs):
        return super()._to_pandas()

# From pandas.DataFrame
def _put_str(s, space):
    return '{s}'.format(s=s)[:space].ljust(space)
