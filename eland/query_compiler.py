import numpy as np
import pandas as pd

from eland import Client
from eland import Index
from eland import Mappings
from eland import Operations


class ElandQueryCompiler:
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

    def __init__(self, client=None, index_pattern=None, field_names=None, index_field=None, operations=None,
                 name_mapper=None):
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

        if field_names is not None:
            self.field_names = field_names

        if name_mapper is None:
            self._name_mapper = ElandQueryCompiler.DisplayNameToFieldNameMapper()
        else:
            self._name_mapper = name_mapper

    def _get_index(self):
        return self._index

    def _get_field_names(self):
        field_names = self._operations.get_field_names()
        if field_names is None:
            # default to all
            field_names = self._mappings.source_fields()

        return pd.Index(field_names)

    def _set_field_names(self, field_names):
        self._operations.set_field_names(field_names)

    field_names = property(_get_field_names, _set_field_names)

    def _get_columns(self):
        columns = self._operations.get_field_names()
        if columns is None:
            # default to all
            columns = self._mappings.source_fields()

        # map renames
        columns = self._name_mapper.field_to_display_names(columns)

        return pd.Index(columns)

    def _set_columns(self, columns):
        # map renames
        columns = self._name_mapper.display_to_field_names(columns)

        self._operations.set_field_names(columns)

    columns = property(_get_columns, _set_columns)

    index = property(_get_index)

    @property
    def dtypes(self):
        columns = self._operations.get_field_names()

        return self._mappings.dtypes(columns)

    # END Index, columns, and dtypes objects

    def _es_results_to_pandas(self, results, batch_size=None):
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

            row = hit['_source']

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
            rows.append(self._flatten_dict(row))

            if batch_size is not None:
                if i >= batch_size:
                    partial_result = True
                    break

        # Create pandas DataFrame
        df = pd.DataFrame(data=rows, index=index)

        # _source may not contain all field_names in the mapping
        # therefore, fill in missing field_names
        # (note this returns self.field_names NOT IN df.columns)
        missing_field_names = list(set(self.field_names) - set(df.columns))

        for missing in missing_field_names:
            is_source_field, pd_dtype = self._mappings.source_field_pd_dtype(missing)
            df[missing] = pd.Series(dtype=pd_dtype)

        # Rename columns
        if not self._name_mapper.empty:
            df.rename(columns=self._name_mapper.display_names_mapper(), inplace=True)

        # Sort columns in mapping order
        if len(self.columns) > 1:
            df = df[self.columns]

        return partial_result, df

    def _flatten_dict(self, y):
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
            elif is_source_field:  # only print source fields from mappings
                # (TODO - not so efficient for large number of fields and filtered mapping)
                field_name = name[:-1]

                # Coerce types - for now just datetime
                if pd_dtype == 'datetime64[ns]':
                    # TODO  - this doesn't work for certain ES date formats
                    #   e.g. "@timestamp" : {
                    #           "type" : "date",
                    #           "format" : "epoch_millis"
                    #         }
                    #   1484053499256 - we need to check ES type and format and add conversions like:
                    #   pd.to_datetime(x, unit='ms')
                    x = pd.to_datetime(x)

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

    def _index_matches(self, items):
        """
        Returns
        -------
        index_count: int
            Count of list of the items that match
        """
        return self._operations.index_matches(self, self.index.index_field, items)

    def _empty_pd_ef(self):
        # Return an empty dataframe with correct columns and dtypes
        df = pd.DataFrame()
        for c, d in zip(self.columns, self.dtypes):
            df[c] = pd.Series(dtype=d)
        return df

    def copy(self):
        return ElandQueryCompiler(client=self._client, index_pattern=self._index_pattern, field_names=None,
                                  index_field=self._index.index_field, operations=self._operations.copy(),
                                  name_mapper=self._name_mapper.copy())

    def rename(self, renames, inplace=False):
        if inplace:
            self._name_mapper.rename_display_name(renames)
            return self
        else:
            result = self.copy()
            result._name_mapper.rename_display_name(renames)
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
    def to_pandas(self):
        """Converts Eland DataFrame to Pandas DataFrame.

        Returns:
            Pandas DataFrame
        """
        return self._operations.to_pandas(self)

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

        result._operations.set_field_names(list(key))

        return result

    def drop(self, index=None, columns=None):
        result = self.copy()

        # Drop gets all columns and removes drops
        if columns is not None:
            # columns is a pandas.Index so we can use pandas drop feature
            new_columns = self.columns.drop(columns)
            result._operations.set_field_names(new_columns.to_list())

        if index is not None:
            result._operations.drop_index_values(self, self.index.index_field, index)

        return result

    def aggs(self, func):
        return self._operations.aggs(self, func)

    def count(self):
        return self._operations.count(self)

    def mean(self):
        return self._operations.mean(self)

    def sum(self):
        return self._operations.sum(self)

    def min(self):
        return self._operations.min(self)

    def max(self):
        return self._operations.max(self)

    def nunique(self):
        return self._operations.nunique(self)

    def value_counts(self, es_size):
        return self._operations.value_counts(self, es_size)

    def info_es(self, buf):
        buf.write("index_pattern: {index_pattern}\n".format(index_pattern=self._index_pattern))

        self._index.info_es(buf)
        self._mappings.info_es(buf)
        self._operations.info_es(buf)

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
        right: ElandQueryCompiler
            The query compiler to compare self to

        Raises
        ------
        TypeError, ValueError
            If arithmetic operations aren't possible
        """
        if not isinstance(right, ElandQueryCompiler):
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

    def check_str_arithmetics(self, right, self_field, right_field):
        """
        In the case of string arithmetics, we need an additional check to ensure that the
        selected fields are aggregatable.

        Parameters
        ----------
        right: ElandQueryCompiler
            The query compiler to compare self to

        Raises
        ------
        TypeError, ValueError
            If string arithmetic operations aren't possible
        """

        # only check compatibility if right is an ElandQueryCompiler
        # else return the raw string as the new field name
        right_agg = {right_field: right_field}
        if right:
            self.check_arithmetics(right)
            right_agg = right._mappings.aggregatable_field_names([right_field])

        self_agg = self._mappings.aggregatable_field_names([self_field])

        if self_agg and right_agg:
            return list(self_agg.keys())[0], list(right_agg.keys())[0]

        else:
            raise ValueError(
                "Can not perform arithmetic operations on non aggregatable fields"
                "One of [{}, {}] is not aggregatable.".format(self.name, right.name)
        )

    def arithmetic_op_fields(self, new_field_name, op, left_field, right_field, op_type=None):
        result = self.copy()

        result._operations.arithmetic_op_fields(new_field_name, op, left_field, right_field, op_type)

        return result

    """
    Internal class to deal with column renaming and script_fields
    """

    class DisplayNameToFieldNameMapper:
        def __init__(self,
                     field_to_display_names=None,
                     display_to_field_names=None):

            if field_to_display_names is not None:
                self._field_to_display_names = field_to_display_names
            else:
                self._field_to_display_names = dict()

            if display_to_field_names is not None:
                self._display_to_field_names = display_to_field_names
            else:
                self._display_to_field_names = dict()

        def rename_display_name(self, renames):
            for current_display_name, new_display_name in renames.items():
                if current_display_name in self._display_to_field_names:
                    # has been renamed already - update name
                    field_name = self._display_to_field_names[current_display_name]
                    del self._display_to_field_names[current_display_name]
                    del self._field_to_display_names[field_name]
                    self._display_to_field_names[new_display_name] = field_name
                    self._field_to_display_names[field_name] = new_display_name
                else:
                    # new rename - assume 'current_display_name' is 'field_name'
                    field_name = current_display_name

                    # if field_name is already mapped ignore
                    if field_name not in self._field_to_display_names:
                        self._display_to_field_names[new_display_name] = field_name
                        self._field_to_display_names[field_name] = new_display_name

        def field_names_to_list(self):
            return sorted(list(self._field_to_display_names.keys()))

        def display_names_to_list(self):
            return sorted(list(self._display_to_field_names.keys()))

        # Return mapper values as dict
        def display_names_mapper(self):
            return self._field_to_display_names

        @property
        def empty(self):
            return not self._display_to_field_names

        def field_to_display_names(self, field_names):
            if self.empty:
                return field_names

            display_names = []

            for field_name in field_names:
                if field_name in self._field_to_display_names:
                    display_name = self._field_to_display_names[field_name]
                else:
                    display_name = field_name
                display_names.append(display_name)

            return display_names

        def display_to_field_names(self, display_names):
            if self.empty:
                return display_names

            field_names = []

            for display_name in display_names:
                if display_name in self._display_to_field_names:
                    field_name = self._display_to_field_names[display_name]
                else:
                    field_name = display_name
                field_names.append(field_name)

            return field_names

        def __constructor__(self, *args, **kwargs):
            return type(self)(*args, **kwargs)

        def copy(self):
            return self.__constructor__(
                field_to_display_names=self._field_to_display_names.copy(),
                display_to_field_names=self._display_to_field_names.copy()
            )
