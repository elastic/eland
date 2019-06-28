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
from elasticsearch_dsl import Search
from pandas.compat import StringIO
from pandas.core import common as com
from pandas.io.common import _expand_user, _stringify_path
from pandas.io.formats import format as fmt
from pandas.io.formats.printing import pprint_thing
from pandas.io.formats import console

import eland as ed


class DataFrame():
    """
    pandas.DataFrame like API that proxies into Elasticsearch index(es).

    Parameters
    ----------
    client : eland.Client
        A reference to a Elasticsearch python client

    index_pattern : str
        An Elasticsearch index pattern. This can contain wildcards (e.g. filebeat-*).

    operations: list of operation
        A list of Elasticsearch analytics operations e.g. filter, aggregations etc.

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

        self._client = ed.Client(client)
        self._index_pattern = index_pattern

        # Get and persist mappings, this allows us to correctly
        # map returned types from Elasticsearch to pandas datatypes
        if mappings is None:
            self._mappings = ed.Mappings(self._client, self._index_pattern)
        else:
            self._mappings = mappings

        self._index = ed.Index(index_field)

    def _es_results_to_pandas(self, results):
        """
        Parameters
        ----------
        results: dict
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
        TODO - explain how lists are handled (https://www.elastic.co/guide/en/elasticsearch/reference/current/array.html)
        TODO - an option here is to use Elasticsearch's multi-field matching instead of pandas treatment of lists (which isn't great)
        NOTE - using this lists is generally not a good way to use this API
        """

        def flatten_dict(y):
            out = {}

            def flatten(x, name=''):
                # We flatten into source fields e.g. if type=geo_point
                # location: {lat=52.38, lon=4.90}
                if name == '':
                    is_source_field = False
                    pd_dtype = 'object'
                else:
                    is_source_field, pd_dtype = self._mappings.source_field_pd_dtype(name[:-1])

                if not is_source_field and type(x) is dict:
                    for a in x:
                        flatten(x[a], name + a + '.')
                elif not is_source_field and type(x) is list:
                    for a in x:
                        flatten(a, name)
                elif is_source_field == True:  # only print source fields from mappings (TODO - not so efficient for large number of fields and filtered mapping)
                    field_name = name[:-1]

                    # Coerce types - for now just datetime
                    if pd_dtype == 'datetime64[ns]':
                        x = pd.to_datetime(x)

                    # Elasticsearch can have multiple values for a field. These are represented as lists, so
                    # create lists for this pivot (see notes above)
                    if field_name in out:
                        if type(out[field_name]) is not list:
                            l = [out[field_name]]
                            out[field_name] = l
                        out[field_name].append(x)
                    else:
                        out[field_name] = x

            flatten(y)

            return out

        rows = []
        index = []
        for hit in results['hits']['hits']:
            row = hit['_source']

            # get index value - can be _id or can be field value in source
            if self._index.is_source_field:
                index_field = row[self._index.index_field]
            else:
                index_field = hit[self._index.index_field]
            index.append(index_field)

            # flatten row to map correctly to 2D DataFrame
            rows.append(flatten_dict(row))

        # Create pandas DataFrame
        df = pd.DataFrame(data=rows, index=index)

        # _source may not contain all columns in the mapping
        # therefore, fill in missing columns
        # (note this returns self.columns NOT IN df.columns)
        missing_columns = list(set(self.columns) - set(df.columns))

        for missing in missing_columns:
            is_source_field, pd_dtype = self._mappings.source_field_pd_dtype(missing)
            df[missing] = None
            df[missing].astype(pd_dtype)

        # Sort columns in mapping order
        df = df[self.columns]

        return df

    def head(self, n=5):
        sort_params = self._index.sort_field + ":asc"

        results = self._client.search(index=self._index_pattern, size=n, sort=sort_params)

        return self._es_results_to_pandas(results)

    def tail(self, n=5):
        sort_params = self._index.sort_field + ":desc"

        results = self._client.search(index=self._index_pattern, size=n, sort=sort_params)

        df = self._es_results_to_pandas(results)

        # reverse order (index ascending)
        return df.sort_index()

    def describe(self):
        numeric_source_fields = self._mappings.numeric_source_fields()

        # for each field we compute:
        # count, mean, std, min, 25%, 50%, 75%, max
        search = Search(using=self._client, index=self._index_pattern).extra(size=0)

        for field in numeric_source_fields:
            search.aggs.metric('extended_stats_' + field, 'extended_stats', field=field)
            search.aggs.metric('percentiles_' + field, 'percentiles', field=field)

        response = search.execute()

        results = {}

        for field in numeric_source_fields:
            values = []
            values.append(response.aggregations['extended_stats_' + field]['count'])
            values.append(response.aggregations['extended_stats_' + field]['avg'])
            values.append(response.aggregations['extended_stats_' + field]['std_deviation'])
            values.append(response.aggregations['extended_stats_' + field]['min'])
            values.append(response.aggregations['percentiles_' + field]['values']['25.0'])
            values.append(response.aggregations['percentiles_' + field]['values']['50.0'])
            values.append(response.aggregations['percentiles_' + field]['values']['75.0'])
            values.append(response.aggregations['extended_stats_' + field]['max'])

            # if not None
            if (values.count(None) < len(values)):
                results[field] = values

        df = pd.DataFrame(data=results, index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])

        return df

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
        lines.append(self.index_summary())

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

    @property
    def columns(self):
        return pd.Index(self._mappings.source_fields())

    @property
    def index(self):
        return self._index

    def set_index(self, index_field):
        copy = self.copy()
        copy._index = ed.Index(index_field)
        return copy

    def index_summary(self):
        head = self.head(1).index[0]
        tail = self.tail(1).index[0]
        index_summary = ', %s to %s' % (pprint_thing(head),
                                        pprint_thing(tail))

        name = "Index"
        return '%s: %s entries%s' % (name, len(self), index_summary)

    @property
    def dtypes(self):
        return self._mappings.dtypes()

    def get_dtype_counts(self):
        return self._mappings.get_dtype_counts()

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

    def index_count(self):
        """
        Returns
        -------
        index_count: int
            Count of docs where index_field exists
        """
        exists_query = {"query": {"exists": {"field": self._index.index_field}}}

        index_count = self._client.count(index=self._index_pattern, body=exists_query)

        return index_count

    def _filter_by_columns(self, columns):
        # Return new eland.DataFrame with modified mappings
        mappings = ed.Mappings(mappings=self._mappings, columns=columns)

        return DataFrame(self._client, self._index_pattern, mappings=mappings)

    def __getitem__(self, key):
        # NOTE: there is a difference between pandas here.
        # e.g. df['a'] returns pd.Series, df[['a','b']] return pd.DataFrame
        # we always return DataFrame - TODO maybe create eland.Series at some point...

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
        if isinstance(key, str):
            if not self._mappings.is_source_field(key):
                raise TypeError('Column does not exist: [{0}]'.format(key))
            columns.append(key)
        elif isinstance(key, list):
            columns.extend(key)
        else:
            raise TypeError('__getitem__ arguments invalid: [{0}]'.format(key))

        return self._filter_by_columns(columns)

    def __len__(self):
        """
        Returns length of info axis, but here we use the index.
        """
        return self._client.count(index=self._index_pattern)

    def copy(self):
        # TODO - test and validate...may need deep copying
        return ed.DataFrame(self._client,
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

        sdf = self.__fake_dataframe__(max_rows=max_rows+1)

        _show_dimensions = show_dimensions

        if buf is not None:
            _buf = _expand_user(_stringify_path(buf))
        else:
            _buf = StringIO()

        sdf.to_string(buf=_buf, columns=columns,
                      col_space=col_space, na_rep=na_rep,
                      formatters=formatters,
                      float_format=float_format,
                      sparsify=sparsify, justify=justify,
                      index_names=index_names,
                      header=header, index=index,
                      max_rows=max_rows,
                      max_cols=max_cols,
                      show_dimensions=False, # print this outside of this call
                      decimal=decimal,
                      line_width=line_width)

        if _show_dimensions:
            _buf.write("\n\n[{nrows} rows x {ncols} columns]"
                      .format(nrows=self.index_count(), ncols=len(self.columns)))

        if buf is None:
            result = _buf.getvalue()
            return result


    def __fake_dataframe__(self, max_rows=1):
        head_rows = int(max_rows / 2) + max_rows % 2
        tail_rows = max_rows - head_rows

        head = self.head(head_rows)
        tail = self.tail(tail_rows)

        num_rows = len(self)

        if (num_rows > max_rows):
            # If we have a lot of rows, create a SparseDataFrame and use
            # pandas to_string logic
            # NOTE: this sparse DataFrame can't be used as the middle
            # section is all NaNs. However, it gives us potentially a nice way
            # to use the pandas IO methods.
            # TODO - if data is indexed by time series, return top/bottom of
            #   time series, rather than first max_rows items
            """
            if tail_rows > 0:
                locations = [0, num_rows - tail_rows]
                lengths = [head_rows, tail_rows]
            else:
                locations = [0]
                lengths = [head_rows]

            sdf = pd.DataFrame({item: pd.SparseArray(data=head[item],
                                                     sparse_index=
                                                     BlockIndex(
                                                         num_rows, locations, lengths))
                                for item in self.columns})
            """
            return pd.concat([head, tail])


        return pd.concat([head, tail])


# From pandas.DataFrame
def _put_str(s, space):
    return '{s}'.format(s=s)[:space].ljust(space)
