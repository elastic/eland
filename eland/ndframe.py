# Copyright 2020 Elasticsearch BV
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from abc import ABC, abstractmethod

from eland.query_compiler import QueryCompiler

"""
NDFrame
---------
Abstract base class for eland.DataFrame and eland.Series.

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


class NDFrame(ABC):
    def __init__(
        self,
        client=None,
        index_pattern=None,
        columns=None,
        index_field=None,
        query_compiler=None,
    ):
        """
        pandas.DataFrame/Series like API that proxies into Elasticsearch index(es).

        Parameters
        ----------
        client : elasticsearch.Elasticsearch
            A reference to a Elasticsearch python client
        """
        if query_compiler is None:
            query_compiler = QueryCompiler(
                client=client,
                index_pattern=index_pattern,
                display_names=columns,
                index_field=index_field,
            )
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
        return self._query_compiler.mean(numeric_only=numeric_only)

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
        return self._query_compiler.sum(numeric_only=numeric_only)

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
        AvgTicketPrice        100.021
        Cancelled               False
        DistanceKilometers          0
        DistanceMiles               0
        FlightDelay             False
        FlightDelayMin              0
        FlightTimeHour              0
        FlightTimeMin               0
        dayOfWeek                   0
        dtype: object
        """
        return self._query_compiler.min(numeric_only=numeric_only)

    def var(self, numeric_only=True):
        """
        Return variance for each numeric column

        Returns
        -------
        pandas.Series
            The value of the variance for each numeric column

        See Also
        --------
        :pandas_api_docs:`pandas.DataFrame.var`

        Examples
        --------
        >>> df = ed.DataFrame('localhost', 'flights')
        >>> df.var() # doctest: +SKIP
        AvgTicketPrice        7.096185e+04
        Cancelled             1.119831e-01
        DistanceKilometers    2.096049e+07
        DistanceMiles         8.092892e+06
        FlightDelay           1.880825e-01
        FlightDelayMin        9.359209e+03
        FlightTimeHour        3.112545e+01
        FlightTimeMin         1.120516e+05
        dayOfWeek             3.761135e+00
        dtype: float64
        """
        return self._query_compiler.var(numeric_only=numeric_only)

    def std(self, numeric_only=True):
        """
        Return standard deviation for each numeric column

        Returns
        -------
        pandas.Series
            The value of the standard deviation for each numeric column

        See Also
        --------
        :pandas_api_docs:`pandas.DataFrame.std`

        Examples
        --------
        >>> df = ed.DataFrame('localhost', 'flights')
        >>> df.std() # doctest: +SKIP
        AvgTicketPrice         266.386661
        Cancelled                0.334639
        DistanceKilometers    4578.263193
        DistanceMiles         2844.800855
        FlightDelay              0.433685
        FlightDelayMin          96.743006
        FlightTimeHour           5.579019
        FlightTimeMin          334.741135
        dayOfWeek                1.939365
        dtype: float64
        """
        return self._query_compiler.std(numeric_only=numeric_only)

    def median(self, numeric_only=True):
        """
        Return the median value for each numeric column

        Returns
        -------
        pandas.Series
            median value for each numeric column

        See Also
        --------
        :pandas_api_docs:`pandas.DataFrame.median`

        Examples
        --------
        >>> df = ed.DataFrame('localhost', 'flights')
        >>> df.median() # doctest: +SKIP
        AvgTicketPrice         640.387285
        Cancelled                0.000000
        DistanceKilometers    7612.072403
        DistanceMiles         4729.922470
        FlightDelay              0.000000
        FlightDelayMin           0.000000
        FlightTimeHour           8.383113
        FlightTimeMin          503.148975
        dayOfWeek                3.000000
        dtype: float64
        """
        return self._query_compiler.median(numeric_only=numeric_only)

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
        AvgTicketPrice        1199.73
        Cancelled                True
        DistanceKilometers    19881.5
        DistanceMiles         12353.8
        FlightDelay              True
        FlightDelayMin            360
        FlightTimeHour         31.715
        FlightTimeMin          1902.9
        dayOfWeek                   6
        dtype: object
        """
        return self._query_compiler.max(numeric_only=numeric_only)

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

    @abstractmethod
    def _to_pandas(self, show_progress=False):
        pass

    @abstractmethod
    def head(self, n=5):
        pass

    @abstractmethod
    def tail(self, n=5):
        pass
