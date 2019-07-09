import pandas as pd
from modin.backends.base.query_compiler import BaseQueryCompiler

from eland import Client
from eland import Index
from eland import Mappings
from eland import Operations

from pandas.core.indexes.numeric import Int64Index
from pandas.core.indexes.range import RangeIndex


class ElandQueryCompiler(BaseQueryCompiler):

    def __init__(self,
                 client=None,
                 index_pattern=None,
                 columns=None,
                 index_field=None,
                 operations=None):
        self._client = Client(client)
        self._index_pattern = index_pattern

        # Get and persist mappings, this allows us to correctly
        # map returned types from Elasticsearch to pandas datatypes
        self._mappings = Mappings(client=self._client, index_pattern=self._index_pattern)

        self._index = Index(self, index_field)

        if operations is None:
            self._operations = Operations()
        else:
            self._operations = operations

        if columns is not None:
            self.columns = columns

    def _get_index(self):
        return self._index

    def _get_columns(self):
        columns = self._operations.get_columns()
        if columns is None:
            # default to all
            columns = self._mappings.source_fields()

        return pd.Index(columns)

    def _set_columns(self, columns):
        self._operations.set_columns(columns)

    columns = property(_get_columns, _set_columns)
    index = property(_get_index)

    # END Index, columns, and dtypes objects

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
        missing_columns = list(set(self.columns) - set(df.columns))

        for missing in missing_columns:
            is_source_field, pd_dtype = self._mappings.source_field_pd_dtype(missing)
            df[missing] = None
            df[missing].astype(pd_dtype)

        # Sort columns in mapping order
        df = df[self.columns]

        return df

    def _index_count(self):
        """
        Returns
        -------
        index_count: int
            Count of docs where index_field exists
        """
        return self._operations.to_count(self)

    def copy(self):
        return self.__constructor__(
            client=self._client,
            index_pattern=self._index_pattern,
            columns=None,   # columns are embedded in operations
            index_field=self._index.index_field,
            operations=self._operations.copy()
        )

    def head(self, n):
        result = self.copy()

        result._operations.head(self._index, n)

        return result

    def tail(self, n):
        result = self.copy()

        result._operations.tail(self._index, n)

        return result

    # To/From Pandas
    def to_pandas(self):
        """Converts Eland DataFrame to Pandas DataFrame.

        Returns:
            Pandas DataFrame
        """
        return self._operations.to_pandas(self)

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

        result._operations.set_columns(key)

        return result

    def squeeze(self, axis=None):
        result = self.copy()

        result._operations.squeeze(axis)

        return result

    def view(self, index=None, columns=None):
        result = self.copy()

        result._operations.iloc(index, columns)

        return result

    def info_es(self, buf):
        buf.write("index_pattern: {index_pattern}\n".format(index_pattern=self._index_pattern))

        self._index.info_es(buf)
        self._mappings.info_es(buf)
        self._operations.info_es(buf)

