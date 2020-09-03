#  Licensed to Elasticsearch B.V. under one or more contributor
#  license agreements. See the NOTICE file distributed with
#  this work for additional information regarding copyright
#  ownership. Elasticsearch B.V. licenses this file to you under
#  the Apache License, Version 2.0 (the "License"); you may
#  not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
# 	http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the License is distributed on an
#  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
#  specific language governing permissions and limitations
#  under the License.

import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Tuple, Optional
import pandas as pd
from eland.query_compiler import QueryCompiler


if TYPE_CHECKING:
    from eland.index import Index

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
        es_client=None,
        es_index_pattern=None,
        columns=None,
        es_index_field=None,
        _query_compiler=None,
    ):
        """
        pandas.DataFrame/Series like API that proxies into Elasticsearch index(es).

        Parameters
        ----------
        client : elasticsearch.Elasticsearch
            A reference to a Elasticsearch python client
        """
        if _query_compiler is None:
            _query_compiler = QueryCompiler(
                client=es_client,
                index_pattern=es_index_pattern,
                display_names=columns,
                index_field=es_index_field,
            )
        self._query_compiler = _query_compiler

    @property
    def index(self) -> "Index":
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
        >>> df.index.es_index_field
        '_id'
        >>> s = df['Carrier']
        >>> assert isinstance(s.index, ed.Index)
        >>> s.index.es_index_field
        '_id'
        """
        return self._query_compiler.index

    @property
    def dtypes(self) -> pd.Series:
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

    def _build_repr(self, num_rows) -> pd.DataFrame:
        # self could be Series or DataFrame
        if len(self.index) <= num_rows:
            return self.to_pandas()

        num_rows = num_rows

        head_rows = int(num_rows / 2) + num_rows % 2
        tail_rows = num_rows - head_rows

        head = self.head(head_rows).to_pandas()
        tail = self.tail(tail_rows).to_pandas()

        return head.append(tail)

    def __sizeof__(self) -> int:
        # Don't default to pandas, just return approximation TODO - make this more accurate
        return sys.getsizeof(self._query_compiler)

    def __len__(self) -> int:
        """Gets the length of the DataFrame.

        Returns:
            Returns an integer length of the DataFrame object.
        """
        return len(self.index)

    def _es_info(self, buf):
        self._query_compiler.es_info(buf)

    def mean(self, numeric_only: Optional[bool] = None) -> pd.Series:
        """
        Return mean value for each numeric column

        TODO - implement remainder of pandas arguments, currently non-numerics are not supported

        Parameters
        ----------
        numeric_only: {True, False, None} Default is None
            Which datatype to be returned
            - True: returns all values with float64, NaN/NaT are ignored.
            - False: returns all values with float64.
            - None: returns all values with default datatypes, NaN/NaT are ignored.

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
        AvgTicketPrice                              628.254
        Cancelled                                      True
        DistanceKilometers                          7092.14
        DistanceMiles                               4406.85
        FlightDelay                                    True
        FlightDelayMin                                   47
        FlightTimeHour                               8.5188
        FlightTimeMin                               511.128
        dayOfWeek                                         2
        timestamp             2018-01-21 19:20:45.564438232
        dtype: object

        >>> df = ed.DataFrame('localhost', 'flights')
        >>> df.mean(numeric_only=True)
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

        >>> df = ed.DataFrame('localhost', 'flights')
        >>> df.mean(numeric_only=False) # doctest: +SKIP
        AvgTicketPrice                              628.254
        Cancelled                                  0.128494
        Carrier                                         NaN
        Dest                                            NaN
        DestAirportID                                   NaN
        DestCityName                                    NaN
        DestCountry                                     NaN
        DestLocation                                    NaN
        DestRegion                                      NaN
        DestWeather                                     NaN
        DistanceKilometers                          7092.14
        DistanceMiles                               4406.85
        FlightDelay                                0.251168
        FlightDelayMin                              47.3352
        FlightDelayType                                 NaN
        FlightNum                                       NaN
        FlightTimeHour                               8.5188
        FlightTimeMin                               511.128
        Origin                                          NaN
        OriginAirportID                                 NaN
        OriginCityName                                  NaN
        OriginCountry                                   NaN
        OriginLocation                                  NaN
        OriginRegion                                    NaN
        OriginWeather                                   NaN
        dayOfWeek                                   2.83598
        timestamp             2018-01-21 19:20:45.564438232
        dtype: object

        """
        return self._query_compiler.mean(numeric_only=numeric_only)

    def sum(self, numeric_only: Optional[bool] = None) -> pd.Series:
        """
        Return sum for each numeric column

        TODO - implement remainder of pandas arguments, currently non-numerics are not supported

        Parameters
        ----------
        numeric_only: {True, False, None} Default is None
            Which datatype to be returned
            - True: returns all values with float64, NaN/NaT are ignored.
            - False: returns all values with float64.
            - None: returns all values with default datatypes, NaN/NaT are ignored.

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
        AvgTicketPrice        8.20436e+06
        Cancelled                    True
        DistanceKilometers    9.26163e+07
        DistanceMiles         5.75491e+07
        FlightDelay                  True
        FlightDelayMin             618150
        FlightTimeHour             111247
        FlightTimeMin         6.67482e+06
        dayOfWeek                   37035
        dtype: object

        >>> df = ed.DataFrame('localhost', 'flights')
        >>> df.sum(numeric_only=True)
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

        >>> df = ed.DataFrame('localhost', 'flights')
        >>> df.sum(numeric_only=False) # doctest: +SKIP
        AvgTicketPrice        8.20436e+06
        Cancelled                    1678
        Carrier                       NaN
        Dest                          NaN
        DestAirportID                 NaN
        DestCityName                  NaN
        DestCountry                   NaN
        DestLocation                  NaN
        DestRegion                    NaN
        DestWeather                   NaN
        DistanceKilometers    9.26163e+07
        DistanceMiles         5.75491e+07
        FlightDelay                  3280
        FlightDelayMin             618150
        FlightDelayType               NaN
        FlightNum                     NaN
        FlightTimeHour             111247
        FlightTimeMin         6.67482e+06
        Origin                        NaN
        OriginAirportID               NaN
        OriginCityName                NaN
        OriginCountry                 NaN
        OriginLocation                NaN
        OriginRegion                  NaN
        OriginWeather                 NaN
        dayOfWeek                   37035
        timestamp                     NaT
        dtype: object

        """
        return self._query_compiler.sum(numeric_only=numeric_only)

    def min(self, numeric_only: Optional[bool] = None) -> pd.Series:
        """
        Return the minimum value for each numeric column

        TODO - implement remainder of pandas arguments, currently non-numerics are not supported

        Parameters
        ----------
        numeric_only: {True, False, None} Default is None
            Which datatype to be returned
            - True: returns all values with float64, NaN/NaT are ignored.
            - False: returns all values with float64.
            - None: returns all values with default datatypes, NaN/NaT are ignored.

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
        AvgTicketPrice                    100.021
        Cancelled                           False
        DistanceKilometers                      0
        DistanceMiles                           0
        FlightDelay                         False
        FlightDelayMin                          0
        FlightTimeHour                          0
        FlightTimeMin                           0
        dayOfWeek                               0
        timestamp             2018-01-01 00:00:00
        dtype: object

        >>> df = ed.DataFrame('localhost', 'flights')
        >>> df.min(numeric_only=True)
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

        >>> df = ed.DataFrame('localhost', 'flights')
        >>> df.min(numeric_only=False) # doctest: +SKIP
        AvgTicketPrice                100.021
        Cancelled                       False
        Carrier                           NaN
        Dest                              NaN
        DestAirportID                     NaN
        DestCityName                      NaN
        DestCountry                       NaN
        DestLocation                      NaN
        DestRegion                        NaN
        DestWeather                       NaN
        DistanceKilometers                  0
        DistanceMiles                       0
        FlightDelay                     False
        FlightDelayMin                      0
        FlightDelayType                   NaN
        FlightNum                         NaN
        FlightTimeHour                      0
        FlightTimeMin                       0
        Origin                            NaN
        OriginAirportID                   NaN
        OriginCityName                    NaN
        OriginCountry                     NaN
        OriginLocation                    NaN
        OriginRegion                      NaN
        OriginWeather                     NaN
        dayOfWeek                           0
        timestamp         2018-01-01 00:00:00
        dtype: object
        """
        return self._query_compiler.min(numeric_only=numeric_only)

    def var(self, numeric_only: Optional[bool] = None) -> pd.Series:
        """
        Return variance for each numeric column

        Parameters
        ----------
        numeric_only: {True, False, None} Default is None
            Which datatype to be returned
            - True: returns all values with float64, NaN/NaT are ignored.
            - False: returns all values with float64.
            - None: returns all values with default datatypes, NaN/NaT are ignored.

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
        >>> df.var()
        AvgTicketPrice            70964.6
        Cancelled                    True
        DistanceKilometers    2.09613e+07
        DistanceMiles          8.0932e+06
        FlightDelay                  True
        FlightDelayMin               9359
        FlightTimeHour            31.1266
        FlightTimeMin              112056
        dayOfWeek                       3
        dtype: object

        >>> df = ed.DataFrame('localhost', 'flights')
        >>> df.var(numeric_only=True)
        AvgTicketPrice        7.096457e+04
        Cancelled             1.119874e-01
        DistanceKilometers    2.096130e+07
        DistanceMiles         8.093202e+06
        FlightDelay           1.880897e-01
        FlightDelayMin        9.359568e+03
        FlightTimeHour        3.112664e+01
        FlightTimeMin         1.120559e+05
        dayOfWeek             3.761279e+00
        dtype: float64

        >>> df = ed.DataFrame('localhost', 'flights')
        >>> df.var(numeric_only=False) # doctest: +SKIP
        AvgTicketPrice            70964.6
        Cancelled                    True
        Carrier                       NaN
        Dest                          NaN
        DestAirportID                 NaN
        DestCityName                  NaN
        DestCountry                   NaN
        DestLocation                  NaN
        DestRegion                    NaN
        DestWeather                   NaN
        DistanceKilometers    2.09613e+07
        DistanceMiles          8.0932e+06
        FlightDelay                  True
        FlightDelayMin               9359
        FlightDelayType               NaN
        FlightNum                     NaN
        FlightTimeHour            31.1266
        FlightTimeMin              112056
        Origin                        NaN
        OriginAirportID               NaN
        OriginCityName                NaN
        OriginCountry                 NaN
        OriginLocation                NaN
        OriginRegion                  NaN
        OriginWeather                 NaN
        dayOfWeek                       3
        timestamp                     NaT
        dtype: object
        """
        return self._query_compiler.var(numeric_only=numeric_only)

    def std(self, numeric_only: Optional[bool] = None) -> pd.Series:
        """
        Return standard deviation for each numeric column

        Parameters
        ----------
        numeric_only: {True, False, None} Default is None
            Which datatype to be returned
            - True: returns all values with float64, NaN/NaT are ignored.
            - False: returns all values with float64.
            - None: returns all values with default datatypes, NaN/NaT are ignored.

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
        >>> df.std()
        AvgTicketPrice        266.407
        Cancelled                True
        DistanceKilometers    4578.61
        DistanceMiles         2845.02
        FlightDelay              True
        FlightDelayMin             96
        FlightTimeHour        5.57945
        FlightTimeMin         334.767
        dayOfWeek                   1
        dtype: object

        >>> df = ed.DataFrame('localhost', 'flights')
        >>> df.std(numeric_only=True)
        AvgTicketPrice         266.407061
        Cancelled                0.334664
        DistanceKilometers    4578.613803
        DistanceMiles         2845.018714
        FlightDelay              0.433718
        FlightDelayMin          96.750415
        FlightTimeHour           5.579446
        FlightTimeMin          334.766770
        dayOfWeek                1.939513
        dtype: float64

        >>> df = ed.DataFrame('localhost', 'flights')
        >>> df.std(numeric_only=False) # doctest: +SKIP
        AvgTicketPrice         266.407
        Cancelled             0.334664
        Carrier                    NaN
        Dest                       NaN
        DestAirportID              NaN
        DestCityName               NaN
        DestCountry                NaN
        DestLocation               NaN
        DestRegion                 NaN
        DestWeather                NaN
        DistanceKilometers     4578.61
        DistanceMiles          2845.02
        FlightDelay           0.433718
        FlightDelayMin         96.7504
        FlightDelayType            NaN
        FlightNum                  NaN
        FlightTimeHour         5.57945
        FlightTimeMin          334.767
        Origin                     NaN
        OriginAirportID            NaN
        OriginCityName             NaN
        OriginCountry              NaN
        OriginLocation             NaN
        OriginRegion               NaN
        OriginWeather              NaN
        dayOfWeek              1.93951
        timestamp                  NaT
        dtype: object

        """
        return self._query_compiler.std(numeric_only=numeric_only)

    def median(self, numeric_only: Optional[bool] = None) -> pd.Series:
        """
        Return the median value for each numeric column

        Parameters
        ----------
        numeric_only: {True, False, None} Default is None
            Which datatype to be returned
            - True: returns all values with float64, NaN/NaT are ignored.
            - False: returns all values with float64.
            - None: returns all values with default datatypes, NaN/NaT are ignored.

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
        AvgTicketPrice                              640.387
        Cancelled                                     False
        DistanceKilometers                          7612.07
        DistanceMiles                               4729.92
        FlightDelay                                   False
        FlightDelayMin                                    0
        FlightTimeHour                              8.38582
        FlightTimeMin                               502.987
        dayOfWeek                                         3
        timestamp             2018-01-21 23:44:29.394281982
        dtype: object

        >>> df = ed.DataFrame('localhost', 'flights')
        >>> df.median(numeric_only=True) # doctest: +SKIP
        AvgTicketPrice         640.424738
        Cancelled                0.000000
        DistanceKilometers    7612.145848
        DistanceMiles         4728.752630
        FlightDelay              0.000000
        FlightDelayMin           0.000000
        FlightTimeHour           8.383711
        FlightTimeMin          502.999492
        dayOfWeek                3.000000
        dtype: float64

        >>> df = ed.DataFrame('localhost', 'flights')
        >>> df.median(numeric_only=False) # doctest: +SKIP
        AvgTicketPrice                              640.425
        Cancelled                                     False
        Carrier                                         NaN
        Dest                                            NaN
        DestAirportID                                   NaN
        DestCityName                                    NaN
        DestCountry                                     NaN
        DestLocation                                    NaN
        DestRegion                                      NaN
        DestWeather                                     NaN
        DistanceKilometers                          7612.15
        DistanceMiles                               4728.75
        FlightDelay                                   False
        FlightDelayMin                                    0
        FlightDelayType                                 NaN
        FlightNum                                       NaN
        FlightTimeHour                              8.38457
        FlightTimeMin                               503.074
        Origin                                          NaN
        OriginAirportID                                 NaN
        OriginCityName                                  NaN
        OriginCountry                                   NaN
        OriginLocation                                  NaN
        OriginRegion                                    NaN
        OriginWeather                                   NaN
        dayOfWeek                                         3
        timestamp             2018-01-21 23:33:14.554215332
        dtype: object

        """
        return self._query_compiler.median(numeric_only=numeric_only)

    def max(self, numeric_only: Optional[bool] = None) -> pd.Series:
        """
        Return the maximum value for each numeric column

        TODO - implement remainder of pandas arguments, currently non-numerics are not supported

        Parameters
        ----------
        numeric_only: {True, False, None} Default is None
            Which datatype to be returned
            - True: returns all values with float64, NaN/NaT are ignored.
            - False: returns all values with float64.
            - None: returns all values with default datatypes, NaN/NaT are ignored.

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
        AvgTicketPrice                    1199.73
        Cancelled                            True
        DistanceKilometers                19881.5
        DistanceMiles                     12353.8
        FlightDelay                          True
        FlightDelayMin                        360
        FlightTimeHour                     31.715
        FlightTimeMin                      1902.9
        dayOfWeek                               6
        timestamp             2018-02-11 23:50:12
        dtype: object

        >>> df = ed.DataFrame('localhost', 'flights')
        >>> df.max(numeric_only=True)
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

        >>> df = ed.DataFrame('localhost', 'flights')
        >>> df.max(numeric_only=False) # doctest: +SKIP
        AvgTicketPrice                1199.73
        Cancelled                        True
        Carrier                           NaN
        Dest                              NaN
        DestAirportID                     NaN
        DestCityName                      NaN
        DestCountry                       NaN
        DestLocation                      NaN
        DestRegion                        NaN
        DestWeather                       NaN
        DistanceKilometers            19881.5
        DistanceMiles                 12353.8
        FlightDelay                      True
        FlightDelayMin                    360
        FlightDelayType                   NaN
        FlightNum                         NaN
        FlightTimeHour                 31.715
        FlightTimeMin                  1902.9
        Origin                            NaN
        OriginAirportID                   NaN
        OriginCityName                    NaN
        OriginCountry                     NaN
        OriginLocation                    NaN
        OriginRegion                      NaN
        OriginWeather                     NaN
        dayOfWeek                           6
        timestamp         2018-02-11 23:50:12
        dtype: object
        """
        return self._query_compiler.max(numeric_only=numeric_only)

    def nunique(self) -> pd.Series:
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

    def mad(self, numeric_only: bool = True) -> pd.Series:
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
        >>> df.mad() # doctest: +SKIP
        AvgTicketPrice         213.368709
        Cancelled                0.000000
        DistanceKilometers    2946.168236
        DistanceMiles         1830.987236
        FlightDelay              0.000000
        FlightDelayMin           0.000000
        FlightTimeHour           3.819435
        FlightTimeMin          229.142297
        dayOfWeek                2.000000
        dtype: float64
        """
        return self._query_compiler.mad(numeric_only=numeric_only)

    def _hist(self, num_bins):
        return self._query_compiler._hist(num_bins)

    def describe(self) -> pd.DataFrame:
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
    def to_pandas(self, show_progress=False):
        pass

    @abstractmethod
    def head(self, n=5):
        pass

    @abstractmethod
    def tail(self, n=5):
        pass

    @abstractmethod
    def sample(self, n=None, frac=None, random_state=None):
        pass

    @property
    def shape(self) -> Tuple[int, ...]:
        raise NotImplementedError

    @property
    def size(self) -> int:
        """
        Return an int representing the number of elements in this object.

        Return the number of rows if Series. Otherwise return the number of rows times number of columns if DataFrame.

        Returns
        -------
        int:
            Number of elements in the object

        See Also
        --------
        :pandas_api_docs:`pandas.DataFrame.size`
        """
        product = 0
        for dim in self.shape:
            product = (product or 1) * dim
        return product
