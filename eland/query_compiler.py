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
import copy
import warnings
from collections import OrderedDict
from datetime import datetime
from typing import Union

import numpy as np
import pandas as pd

from eland import Client, DEFAULT_PROGRESS_REPORTING_NUM_ROWS, elasticsearch_date_to_pandas_date
from eland import FieldMappings
from eland import Index
from eland import Operations


class QueryCompiler:
    """
    Some notes on what can and can not be mapped:

    1. df.head(10)

    /_search?size=10

    2. df.tail(10)

    /_search?size=10&sort=_doc:desc
    + post_process results (sort_index)

    3. df[['OriginAirportID', 'AvgTicketPrice', 'Carrier']]

    /_search
    { '_source': ['OriginAirportID', 'AvgTicketPrice', 'Carrier']}

    4. df.drop(['1', '2'])

    /_search
    {'query': {'bool': {'must': [], 'must_not': [{'ids': {'values': ['1', '2']}}]}}, 'aggs': {}}

    This doesn't work is size is set (e.g. head/tail) as we don't know in Elasticsearch if values '1' or '2' are
    in the first/last n fields.

    A way to mitigate this would be to post process this drop - TODO
    """

    def __init__(self,
                 client=None,
                 index_pattern=None,
                 display_names=None,
                 index_field=None,
                 to_copy=None):
        # Implement copy as we don't deep copy the client
        if to_copy is not None:
            self._client = Client(to_copy._client)
            self._index_pattern = to_copy._index_pattern
            self._index = Index(self, to_copy._index.index_field)
            self._operations = copy.deepcopy(to_copy._operations)
            self._mappings = copy.deepcopy(to_copy._mappings)
        else:
            self._client = Client(client)
            self._index_pattern = index_pattern
            # Get and persist mappings, this allows us to correctly
            # map returned types from Elasticsearch to pandas datatypes
            self._mappings = FieldMappings(client=self._client, index_pattern=self._index_pattern,
                                           display_names=display_names)
            self._index = Index(self, index_field)
            self._operations = Operations()

    @property
    def index(self):
        return self._index

    @property
    def columns(self):
        columns = self._mappings.display_names

        return pd.Index(columns)

    def _get_display_names(self):
        display_names = self._mappings.display_names

        return pd.Index(display_names)

    def _set_display_names(self, display_names):
        self._mappings.display_names = display_names

    def get_field_names(self, include_scripted_fields):
        return self._mappings.get_field_names(include_scripted_fields)

    def add_scripted_field(self, scripted_field_name, display_name, pd_dtype):
        result = self.copy()
        self._mappings.add_scripted_field(scripted_field_name, display_name, pd_dtype)
        return result

    @property
    def dtypes(self):
        return self._mappings.dtypes()

    # END Index, columns, and dtypes objects

    def _es_results_to_pandas(self, results, batch_size=None, show_progress=False):
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
        TODO - explain how lists are handled
            (https://www.elastic.co/guide/en/elasticsearch/reference/current/array.html)
        TODO - an option here is to use Elasticsearch's multi-field matching instead of pandas treatment of lists
            (which isn't great)
        NOTE - using this lists is generally not a good way to use this API
        """
        partial_result = False

        if results is None:
            return partial_result, self._empty_pd_ef()

        # This is one of the most performance critical areas of eland, and it repeatedly calls
        # self._mappings.field_name_pd_dtype and self._mappings.date_field_format
        # therefore create a simple cache for this data
        field_mapping_cache = FieldMappingCache(self._mappings)

        rows = []
        index = []
        if isinstance(results, dict):
            iterator = results['hits']['hits']

            if batch_size is not None:
                raise NotImplementedError("Can not specify batch_size with dict results")
        else:
            iterator = results

        i = 0
        for hit in iterator:
            i = i + 1

            if '_source' in hit:
                row = hit['_source']
            else:
                row = {}

            # script_fields appear in 'fields'
            if 'fields' in hit:
                fields = hit['fields']
                for key, value in fields.items():
                    row[key] = value

            # get index value - can be _id or can be field value in source
            if self._index.is_source_field:
                index_field = row[self._index.index_field]
            else:
                index_field = hit[self._index.index_field]
            index.append(index_field)

            # flatten row to map correctly to 2D DataFrame
            rows.append(self._flatten_dict(row, field_mapping_cache))

            if batch_size is not None:
                if i >= batch_size:
                    partial_result = True
                    break

            if show_progress:
                if i % DEFAULT_PROGRESS_REPORTING_NUM_ROWS == 0:
                    print("{}: read {} rows".format(datetime.now(), i))

        # Create pandas DataFrame
        df = pd.DataFrame(data=rows, index=index)

        # _source may not contain all field_names in the mapping
        # therefore, fill in missing field_names
        # (note this returns self.field_names NOT IN df.columns)
        missing_field_names = list(set(self.get_field_names(include_scripted_fields=True)) - set(df.columns))

        for missing in missing_field_names:
            pd_dtype = self._mappings.field_name_pd_dtype(missing)
            df[missing] = pd.Series(dtype=pd_dtype)

        # Rename columns
        df.rename(columns=self._mappings.get_renames(), inplace=True)

        # Sort columns in mapping order
        if len(self.columns) > 1:
            df = df[self.columns]

        if show_progress:
            print("{}: read {} rows".format(datetime.now(), i))

        return partial_result, df

    def _flatten_dict(self, y, field_mapping_cache):
        out = OrderedDict()

        def flatten(x, name=''):
            # We flatten into source fields e.g. if type=geo_point
            # location: {lat=52.38, lon=4.90}
            if name == '':
                is_source_field = False
                pd_dtype = 'object'
            else:
                try:
                    pd_dtype = field_mapping_cache.field_name_pd_dtype(name[:-1])
                    is_source_field = True
                except KeyError:
                    is_source_field = False
                    pd_dtype = 'object'

            if not is_source_field and type(x) is dict:
                for a in x:
                    flatten(x[a], name + a + '.')
            elif not is_source_field and type(x) is list:
                for a in x:
                    flatten(a, name)
            elif is_source_field:  # only print source fields from mappings
                # (TODO - not so efficient for large number of fields and filtered mapping)
                field_name = name[:-1]

                # Coerce types - for now just datetime
                if pd_dtype == 'datetime64[ns]':
                    x = elasticsearch_date_to_pandas_date(
                        x,
                        field_mapping_cache.date_field_format(field_name)
                    )

                # Elasticsearch can have multiple values for a field. These are represented as lists, so
                # create lists for this pivot (see notes above)
                if field_name in out:
                    if type(out[field_name]) is not list:
                        field_as_list = [out[field_name]]
                        out[field_name] = field_as_list
                    out[field_name].append(x)
                else:
                    out[field_name] = x
            else:
                # Script fields end up here

                # Elasticsearch returns 'Infinity' as a string for np.inf values.
                # Map this to a numeric value to avoid this whole Series being classed as an object
                # TODO - create a lookup for script fields and dtypes to only map 'Infinity'
                #        if the field is numeric. This implementation will currently map
                #        any script field with "Infinity" as a string to np.inf
                if x == 'Infinity':
                    out[name[:-1]] = np.inf
                else:
                    out[name[:-1]] = x

        flatten(y)

        return out

    def _index_count(self):
        """
        Returns
        -------
        index_count: int
            Count of docs where index_field exists
        """
        return self._operations.index_count(self, self.index.index_field)

    def _index_matches_count(self, items):
        """
        Returns
        -------
        index_count: int
            Count of docs where items exist
        """
        return self._operations.index_matches_count(self, self.index.index_field, items)

    def _empty_pd_ef(self):
        # Return an empty dataframe with correct columns and dtypes
        df = pd.DataFrame()
        for c, d in zip(self.dtypes.index, self.dtypes.values):
            df[c] = pd.Series(dtype=d)
        return df

    def copy(self):
        return QueryCompiler(to_copy=self)

    def rename(self, renames, inplace=False):
        if inplace:
            self._mappings.rename(renames)
            return self
        else:
            result = self.copy()
            result._mappings.rename(renames)
            return result

    def head(self, n):
        result = self.copy()

        result._operations.head(self._index, n)

        return result

    def tail(self, n):
        result = self.copy()

        result._operations.tail(self._index, n)

        return result

    # To/From Pandas
    def to_pandas(self, show_progress=False):
        """Converts Eland DataFrame to Pandas DataFrame.

        Returns:
            Pandas DataFrame
        """
        return self._operations.to_pandas(self, show_progress)

    # To CSV
    def to_csv(self, **kwargs):
        """Serialises Eland Dataframe to CSV

        Returns:
            If path_or_buf is None, returns the resulting csv format as a string. Otherwise returns None.
        """
        return self._operations.to_csv(self, **kwargs)

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

        result._mappings.display_names = list(key)

        return result

    def drop(self, index=None, columns=None):
        result = self.copy()

        # Drop gets all columns and removes drops
        if columns is not None:
            # columns is a pandas.Index so we can use pandas drop feature
            new_columns = self.columns.drop(columns)
            result._mappings.display_names = new_columns.to_list()

        if index is not None:
            result._operations.drop_index_values(self, self.index.index_field, index)

        return result

    def aggs(self, func):
        return self._operations.aggs(self, func)

    def count(self):
        return self._operations.count(self)

    def mean(self, numeric_only=None):
        return self._operations.mean(self, numeric_only=numeric_only)

    def sum(self, numeric_only=None):
        return self._operations.sum(self, numeric_only=numeric_only)

    def min(self, numeric_only=None):
        return self._operations.min(self, numeric_only=numeric_only)

    def max(self, numeric_only=None):
        return self._operations.max(self, numeric_only=numeric_only)

    def nunique(self):
        return self._operations.nunique(self)

    def value_counts(self, es_size):
        return self._operations.value_counts(self, es_size)

    def info_es(self, buf):
        buf.write("index_pattern: {index_pattern}\n".format(index_pattern=self._index_pattern))

        self._index.info_es(buf)
        self._mappings.info_es(buf)
        self._operations.info_es(self, buf)

    def describe(self):
        return self._operations.describe(self)

    def _hist(self, num_bins):
        return self._operations.hist(self, num_bins)

    def _update_query(self, boolean_filter):
        result = self.copy()

        result._operations.update_query(boolean_filter)

        return result

    def check_arithmetics(self, right):
        """
        Compare 2 query_compilers to see if arithmetic operations can be performed by the NDFrame object.

        This does very basic comparisons and ignores some of the complexities of incompatible task lists

        Raises exception if incompatible

        Parameters
        ----------
        right: QueryCompiler
            The query compiler to compare self to

        Raises
        ------
        TypeError, ValueError
            If arithmetic operations aren't possible
        """
        if not isinstance(right, QueryCompiler):
            raise TypeError(
                "Incompatible types "
                "{0} != {1}".format(type(self), type(right))
            )

        if self._client._es != right._client._es:
            raise ValueError(
                "Can not perform arithmetic operations across different clients"
                "{0} != {1}".format(self._client._es, right._client._es)
            )

        if self._index.index_field != right._index.index_field:
            raise ValueError(
                "Can not perform arithmetic operations across different index fields "
                "{0} != {1}".format(self._index.index_field, right._index.index_field)
            )

        if self._index_pattern != right._index_pattern:
            raise ValueError(
                "Can not perform arithmetic operations across different index patterns"
                "{0} != {1}".format(self._index_pattern, right._index_pattern)
            )

    def arithmetic_op_fields(self, display_name, arithmetic_object):
        result = self.copy()

        # create a new field name for this display name
        scripted_field_name = "script_field_{}".format(display_name)

        # add scripted field
        result._mappings.add_scripted_field(scripted_field_name, display_name, arithmetic_object.dtype.name)

        result._operations.arithmetic_op_fields(scripted_field_name, arithmetic_object)

        return result

    def get_arithmetic_op_fields(self):
        return self._operations.get_arithmetic_op_fields()

    def display_name_to_aggregatable_name(self, display_name):
        aggregatable_field_name = self._mappings.aggregatable_field_name(display_name)

        return aggregatable_field_name

class FieldMappingCache:
    """
    Very simple dict cache for field mappings. This improves performance > 3 times on large datasets as
    DataFrame access is slower than dict access.
    """

    def __init__(self, mappings):
        self._mappings = mappings

        self._field_name_pd_dtype = dict()
        self._date_field_format = dict()

    def field_name_pd_dtype(self, es_field_name):
        if es_field_name in self._field_name_pd_dtype:
            return self._field_name_pd_dtype[es_field_name]

        pd_dtype = self._mappings.field_name_pd_dtype(es_field_name)

        # cache this
        self._field_name_pd_dtype[es_field_name] = pd_dtype

        return pd_dtype

    def date_field_format(self, es_field_name):
        if es_field_name in self._date_field_format:
            return self._date_field_format[es_field_name]

        es_date_field_format = self._mappings.date_field_format(es_field_name)

        # cache this
        self._date_field_format[es_field_name] = es_date_field_format

        return es_date_field_format
