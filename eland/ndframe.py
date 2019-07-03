"""
NDFrame
---------
Base class for eland.DataFrame and eland.Series.

The underlying data resides in Elasticsearch and the API aligns as much as
possible with pandas APIs.

This allows the eland.DataFrame to access large datasets stored in Elasticsearch,
without storing the dataset in local memory.

Implementation Details
----------------------

Elasticsearch indexes can be configured in many different ways, and these indexes
utilise different data structures to pandas.

eland.DataFrame operations that return individual rows (e.g. df.head()) return
_source data. If _source is not enabled, this data is not accessible.

Similarly, only Elasticsearch searchable fields can be searched or filtered, and
only Elasticsearch aggregatable fields can be aggregated or grouped.

"""
import pandas as pd
import functools
from elasticsearch_dsl import Search

import eland as ed

from pandas.core.generic import NDFrame as pd_NDFrame
from pandas._libs import Timestamp, iNaT, properties


class NDFrame():
    """
    pandas.DataFrame/Series like API that proxies into Elasticsearch index(es).

    Parameters
    ----------
    client : eland.Client
        A reference to a Elasticsearch python client

    index_pattern : str
        An Elasticsearch index pattern. This can contain wildcards (e.g. filebeat-*).

    See Also
    --------

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
        if isinstance(results, dict):
            iterator = results['hits']['hits']
        else:
            iterator = results

        for hit in iterator:
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
        missing_columns = list(set(self._columns) - set(df.columns))

        for missing in missing_columns:
            is_source_field, pd_dtype = self._mappings.source_field_pd_dtype(missing)
            df[missing] = None
            df[missing].astype(pd_dtype)

        # Sort columns in mapping order
        df = df[self._columns]

        return df

    def _head(self, n=5):
        """
        Protected method that returns head as pandas.DataFrame.

        Returns
        -------
        _head
            pandas.DataFrame of top N values
        """
        sort_params = self._index.sort_field + ":asc"

        results = self._client.search(index=self._index_pattern, size=n, sort=sort_params)

        return self._es_results_to_pandas(results)

    def _tail(self, n=5):
        """
        Protected method that returns tail as pandas.DataFrame.

        Returns
        -------
        _tail
            pandas.DataFrame of last N values
        """
        sort_params = self._index.sort_field + ":desc"

        results = self._client.search(index=self._index_pattern, size=n, sort=sort_params)

        df = self._es_results_to_pandas(results)

        # reverse order (index ascending)
        return df.sort_index()

    def _to_pandas(self):
        """
        Protected method that returns all data as pandas.DataFrame.

        Returns
        -------
        df
            pandas.DataFrame of all values
        """
        sort_params = self._index.sort_field + ":asc"

        results = self._client.scan(index=self._index_pattern)

        # We sort here rather than in scan - once everything is in core this
        # should be faster
        return self._es_results_to_pandas(results)

    def _describe(self):
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
            values = list()
            values.append(response.aggregations['extended_stats_' + field]['count'])
            values.append(response.aggregations['extended_stats_' + field]['avg'])
            values.append(response.aggregations['extended_stats_' + field]['std_deviation'])
            values.append(response.aggregations['extended_stats_' + field]['min'])
            values.append(response.aggregations['percentiles_' + field]['values']['25.0'])
            values.append(response.aggregations['percentiles_' + field]['values']['50.0'])
            values.append(response.aggregations['percentiles_' + field]['values']['75.0'])
            values.append(response.aggregations['extended_stats_' + field]['max'])

            # if not None
            if values.count(None) < len(values):
                results[field] = values

        df = pd.DataFrame(data=results, index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])

        return df

    def _filter_mappings(self, columns):
        mappings = ed.Mappings(mappings=self._mappings, columns=columns)

        return mappings

    @property
    def columns(self):
        return self._columns

    @property
    def index(self):
        return self._index

    @property
    def dtypes(self):
        return self._mappings.dtypes()

    @property
    def _columns(self):
        return pd.Index(self._mappings.source_fields())

    def get_dtype_counts(self):
        return self._mappings.get_dtype_counts()

    def _index_count(self):
        """
        Returns
        -------
        index_count: int
            Count of docs where index_field exists
        """
        exists_query = {"query": {"exists": {"field": self._index.index_field}}}

        index_count = self._client.count(index=self._index_pattern, body=exists_query)

        return index_count

    def __len__(self):
        """
        Returns length of info axis, but here we use the index.
        """
        return self._client.count(index=self._index_pattern)

    def _fake_head_tail_df(self, max_rows=1):
        """
        Create a 'fake' pd.DataFrame of the entire ed.DataFrame
        by concat head and tail. Used for display.
        """
        head_rows = int(max_rows / 2) + max_rows % 2
        tail_rows = max_rows - head_rows

        head = self._head(head_rows)
        tail = self._tail(tail_rows)

        return head.append(tail)
