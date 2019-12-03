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

import sys

import pandas as pd
from pandas.core.dtypes.common import is_list_like
from pandas.util._validators import validate_bool_kwarg

from eland import ElandQueryCompiler


class NDFrame:

    def __init__(self,
                 client=None,
                 index_pattern=None,
                 columns=None,
                 index_field=None,
                 query_compiler=None):
        """
        pandas.DataFrame/Series like API that proxies into Elasticsearch index(es).

        Parameters
        ----------
        client : eland.Client
            A reference to a Elasticsearch python client
        """
        if query_compiler is None:
            query_compiler = ElandQueryCompiler(client=client, index_pattern=index_pattern, field_names=columns,
                                                index_field=index_field)
        self._query_compiler = query_compiler

    def _get_index(self):
        """
        Return eland index referencing Elasticsearch field to index a DataFrame/Series

        Returns
        -------
        eland.Index:
            Note eland.Index has a very limited API compared to pandas.Index

        See Also
        --------
        :pandas_api_docs:`pandas.DataFrame.index`
        :pandas_api_docs:`pandas.Series.index`

        Examples
        --------
        >>> df = ed.DataFrame('localhost', 'flights')
        >>> assert isinstance(df.index, ed.Index)
        >>> df.index.index_field
        '_id'
        >>> s = df['Carrier']
        >>> assert isinstance(s.index, ed.Index)
        >>> s.index.index_field
        '_id'
        """
        return self._query_compiler.index

    index = property(_get_index)

    @property
    def dtypes(self):
        """
        Return the pandas dtypes in the DataFrame. Elasticsearch types are mapped
        to pandas dtypes via Mappings._es_dtype_to_pd_dtype.__doc__

        Returns
        -------
        pandas.Series
            The data type of each column.

        See Also
        --------
        :pandas_api_docs:`pandas.DataFrame.dtypes`

        Examples
        --------
        >>> df = ed.DataFrame('localhost', 'flights', columns=['Origin', 'AvgTicketPrice', 'timestamp', 'dayOfWeek'])
        >>> df.dtypes
        Origin                    object
        AvgTicketPrice           float64
        timestamp         datetime64[ns]
        dayOfWeek                  int64
        dtype: object
        """
        return self._query_compiler.dtypes

    def _build_repr(self, num_rows):
        # self could be Series or DataFrame
        if len(self.index) <= num_rows:
            return self._to_pandas()

        num_rows = num_rows

        head_rows = int(num_rows / 2) + num_rows % 2
        tail_rows = num_rows - head_rows

        head = self.head(head_rows)._to_pandas()
        tail = self.tail(tail_rows)._to_pandas()

        return head.append(tail)

    def __getitem__(self, key):
        return self._getitem(key)

    def __sizeof__(self):
        # Don't default to pandas, just return approximation TODO - make this more accurate
        return sys.getsizeof(self._query_compiler)

    def __len__(self):
        """Gets the length of the DataFrame.

        Returns:
            Returns an integer length of the DataFrame object.
        """
        return len(self.index)

    def _info_es(self, buf):
        self._query_compiler.info_es(buf)

    def drop(
            self,
            labels=None,
            axis=0,
            index=None,
            columns=None,
            level=None,
            inplace=False,
            errors="raise",
    ):
        """Return new object with labels in requested axis removed.

        Parameters
        ----------
        labels:
            Index or column labels to drop.
        axis:
            Whether to drop labels from the index (0 / 'index') or columns (1 / 'columns').
        index, columns:
            Alternative to specifying axis (labels, axis=1 is equivalent to columns=labels).
        level:
            For MultiIndex - not supported
        inplace:
            If True, do operation inplace and return None.
        errors:
            If 'ignore', suppress error and existing labels are dropped.

        Returns
        -------
        dropped:
            type of caller

        See Also
        --------
        :pandas_api_docs:`pandas.DataFrame.drop`

        Examples
        --------
        Drop a column

        >>> df = ed.DataFrame('localhost', 'ecommerce', columns=['customer_first_name', 'email', 'user'])
        >>> df.drop(columns=['user'])
             customer_first_name                       email
        0                  Eddie  eddie@underwood-family.zzz
        1                   Mary      mary@bailey-family.zzz
        2                   Gwen      gwen@butler-family.zzz
        3                  Diane   diane@chandler-family.zzz
        4                  Eddie      eddie@weber-family.zzz
        ...                  ...                         ...
        4670                Mary     mary@lambert-family.zzz
        4671                 Jim      jim@gilbert-family.zzz
        4672               Yahya     yahya@rivera-family.zzz
        4673                Mary     mary@hampton-family.zzz
        4674             Jackson  jackson@hopkins-family.zzz
        <BLANKLINE>
        [4675 rows x 2 columns]

        Drop rows by index value (axis=0)

        >>> df.drop(['1', '2'])
             customer_first_name                       email     user
        0                  Eddie  eddie@underwood-family.zzz    eddie
        3                  Diane   diane@chandler-family.zzz    diane
        4                  Eddie      eddie@weber-family.zzz    eddie
        5                  Diane    diane@goodwin-family.zzz    diane
        6                 Oliver      oliver@rios-family.zzz   oliver
        ...                  ...                         ...      ...
        4670                Mary     mary@lambert-family.zzz     mary
        4671                 Jim      jim@gilbert-family.zzz      jim
        4672               Yahya     yahya@rivera-family.zzz    yahya
        4673                Mary     mary@hampton-family.zzz     mary
        4674             Jackson  jackson@hopkins-family.zzz  jackson
        <BLANKLINE>
        [4673 rows x 3 columns]
        """
        # Level not supported
        if level is not None:
            raise NotImplementedError("level not supported {}".format(level))

        inplace = validate_bool_kwarg(inplace, "inplace")
        if labels is not None:
            if index is not None or columns is not None:
                raise ValueError("Cannot specify both 'labels' and 'index'/'columns'")
            axis = pd.DataFrame()._get_axis_name(axis)
            axes = {axis: labels}
        elif index is not None or columns is not None:
            axes, _ = pd.DataFrame()._construct_axes_from_arguments(
                (index, columns), {}
            )
        else:
            raise ValueError(
                "Need to specify at least one of 'labels', 'index' or 'columns'"
            )

        # TODO Clean up this error checking
        if "index" not in axes:
            axes["index"] = None
        elif axes["index"] is not None:
            if not is_list_like(axes["index"]):
                axes["index"] = [axes["index"]]
            if errors == "raise":
                # Check if axes['index'] values exists in index
                count = self._query_compiler._index_matches_count(axes["index"])
                if count != len(axes["index"]):
                    raise ValueError(
                        "number of labels {}!={} not contained in axis".format(count, len(axes["index"]))
                    )
            else:
                """
                axes["index"] = self._query_compiler.index_matches(axes["index"])
                # If the length is zero, we will just do nothing
                if not len(axes["index"]):
                    axes["index"] = None
                """
                raise NotImplementedError()

        if "columns" not in axes:
            axes["columns"] = None
        elif axes["columns"] is not None:
            if not is_list_like(axes["columns"]):
                axes["columns"] = [axes["columns"]]
            if errors == "raise":
                non_existant = [
                    obj for obj in axes["columns"] if obj not in self.columns
                ]
                if len(non_existant):
                    raise ValueError(
                        "labels {} not contained in axis".format(non_existant)
                    )
            else:
                axes["columns"] = [
                    obj for obj in axes["columns"] if obj in self.columns
                ]
                # If the length is zero, we will just do nothing
                if not len(axes["columns"]):
                    axes["columns"] = None

        new_query_compiler = self._query_compiler.drop(
            index=axes["index"], columns=axes["columns"]
        )
        return self._create_or_update_from_compiler(new_query_compiler, inplace)

    def mean(self, numeric_only=True):
        """
        Return mean value for each numeric column

        TODO - implement remainder of pandas arguments, currently non-numerics are not supported

        Returns
        -------
        pandas.Series
            mean value for each numeric column

        See Also
        --------
        :pandas_api_docs:`pandas.DataFrame.mean`

        Examples
        --------
        >>> df = ed.DataFrame('localhost', 'flights')
        >>> df.mean()
        AvgTicketPrice         628.253689
        Cancelled                0.128494
        DistanceKilometers    7092.142457
        DistanceMiles         4406.853010
        FlightDelay              0.251168
        FlightDelayMin          47.335171
        FlightTimeHour           8.518797
        FlightTimeMin          511.127842
        dayOfWeek                2.835975
        dtype: float64
        """
        if not numeric_only:
            raise NotImplementedError("Only mean of numeric fields is implemented")
        return self._query_compiler.mean()

    def sum(self, numeric_only=True):
        """
        Return sum for each numeric column

        TODO - implement remainder of pandas arguments, currently non-numerics are not supported

        Returns
        -------
        pandas.Series
            sum for each numeric column

        See Also
        --------
        :pandas_api_docs:`pandas.DataFrame.sum`

        Examples
        --------
        >>> df = ed.DataFrame('localhost', 'flights')
        >>> df.sum()
        AvgTicketPrice        8.204365e+06
        Cancelled             1.678000e+03
        DistanceKilometers    9.261629e+07
        DistanceMiles         5.754909e+07
        FlightDelay           3.280000e+03
        FlightDelayMin        6.181500e+05
        FlightTimeHour        1.112470e+05
        FlightTimeMin         6.674818e+06
        dayOfWeek             3.703500e+04
        dtype: float64
        """
        if not numeric_only:
            raise NotImplementedError("Only sum of numeric fields is implemented")
        return self._query_compiler.sum()

    def min(self, numeric_only=True):
        """
        Return the minimum value for each numeric column

        TODO - implement remainder of pandas arguments, currently non-numerics are not supported

        Returns
        -------
        pandas.Series
            min value for each numeric column

        See Also
        --------
        :pandas_api_docs:`pandas.DataFrame.min`

        Examples
        --------
        >>> df = ed.DataFrame('localhost', 'flights')
        >>> df.min()
        AvgTicketPrice        100.020531
        Cancelled               0.000000
        DistanceKilometers      0.000000
        DistanceMiles           0.000000
        FlightDelay             0.000000
        FlightDelayMin          0.000000
        FlightTimeHour          0.000000
        FlightTimeMin           0.000000
        dayOfWeek               0.000000
        dtype: float64
        """
        if not numeric_only:
            raise NotImplementedError("Only min of numeric fields is implemented")
        return self._query_compiler.min()

    def max(self, numeric_only=True):
        """
        Return the maximum value for each numeric column

        TODO - implement remainder of pandas arguments, currently non-numerics are not supported

        Returns
        -------
        pandas.Series
            max value for each numeric column

        See Also
        --------
        :pandas_api_docs:`pandas.DataFrame.max`

        Examples
        --------
        >>> df = ed.DataFrame('localhost', 'flights')
        >>> df.max()
        AvgTicketPrice         1199.729004
        Cancelled                 1.000000
        DistanceKilometers    19881.482422
        DistanceMiles         12353.780273
        FlightDelay               1.000000
        FlightDelayMin          360.000000
        FlightTimeHour           31.715034
        FlightTimeMin          1902.901978
        dayOfWeek                 6.000000
        dtype: float64
        """
        if not numeric_only:
            raise NotImplementedError("Only max of numeric fields is implemented")
        return self._query_compiler.max()

    def nunique(self):
        """
        Return cardinality of each field.

        **Note we can only do this for aggregatable Elasticsearch fields - (in general) numeric and keyword
        rather than text fields**

        This method will try and field aggregatable fields if possible if mapping has::

            "customer_first_name" : {
              "type" : "text",
              "fields" : {
                "keyword" : {
                  "type" : "keyword",
                  "ignore_above" : 256
                }
              }
            }

        we will aggregate ``customer_first_name`` columns using ``customer_first_name.keyword``.

        TODO - implement remainder of pandas arguments

        Returns
        -------
        pandas.Series
            cardinality of each column

        See Also
        --------
        :pandas_api_docs:`pandas.DataFrame.nunique`

        Examples
        --------
        >>> columns = ['category', 'currency', 'customer_birth_date', 'customer_first_name', 'user']
        >>> df = ed.DataFrame('localhost', 'ecommerce', columns=columns)
        >>> df.nunique()
        category                6
        currency                1
        customer_birth_date     0
        customer_first_name    46
        user                   46
        dtype: int64
        """
        return self._query_compiler.nunique()

    def _hist(self, num_bins):
        return self._query_compiler._hist(num_bins)

    def describe(self):
        """
        Generate descriptive statistics that summarize the central tendency, dispersion and shape of a
        dataset’s distribution, excluding NaN values.

        Analyzes both numeric and object series, as well as DataFrame column sets of mixed data types.
        The output will vary depending on what is provided. Refer to the notes below for more detail.

        TODO - add additional arguments (current only numeric values supported)

        Returns
        -------
        pandas.Dataframe:
            Summary information

        See Also
        --------
        :pandas_api_docs:`pandas.DataFrame.describe`

        Examples
        --------
        >>> df = ed.DataFrame('localhost', 'flights', columns=['AvgTicketPrice', 'FlightDelayMin'])
        >>> df.describe() # ignoring percentiles as they don't generate consistent results
               AvgTicketPrice  FlightDelayMin
        count    13059.000000    13059.000000
        mean       628.253689       47.335171
        std        266.386661       96.743006
        min        100.020531        0.000000
        ...
        ...
        ...
        max       1199.729004      360.000000
        """
        return self._query_compiler.describe()
