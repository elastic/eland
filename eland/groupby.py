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

from typing import TYPE_CHECKING, List, Optional, Union

from eland.query_compiler import QueryCompiler

if TYPE_CHECKING:
    import pandas as pd  # type: ignore


class GroupBy:
    """
    Base class for calls to :py:func:`eland.DataFrame.groupby`
    """

    def __init__(
        self,
        by: List[str],
        query_compiler: "QueryCompiler",
        dropna: bool = True,
    ) -> None:
        self._query_compiler: "QueryCompiler" = QueryCompiler(to_copy=query_compiler)
        self._dropna: bool = dropna
        self._by: List[str] = by


class DataFrameGroupBy(GroupBy):
    """
    This holds all the groupby methods for :py:func:`eland.DataFrame.groupby`
    """

    def mean(self, numeric_only: bool = True) -> "pd.DataFrame":
        """
        Compute the mean value for each group.

        Parameters
        ----------
        numeric_only: {True, False, None} Default is True
            Which datatype to be returned
            - True: Returns all values as float64, NaN/NaT values are removed
            - None: Returns all values as the same dtype where possible, NaN/NaT are removed
            - False: Returns all values as the same dtype where possible, NaN/NaT are preserved

        Returns
        -------
        pandas.DataFrame
            mean value for each numeric column of each group

        See Also
        --------
        :pandas_api_docs:`pandas.core.groupby.GroupBy.mean`

        Examples
        --------
        >>> df = ed.DataFrame(
        ...   "localhost", "flights",
        ...   columns=["AvgTicketPrice", "Cancelled", "dayOfWeek", "timestamp", "DestCountry"]
        ... )
        >>> df.groupby("DestCountry").mean(numeric_only=False) # doctest: +SKIP
                     AvgTicketPrice  Cancelled  dayOfWeek                     timestamp
        DestCountry
        AE               605.132970   0.152174   2.695652 2018-01-21 16:58:07.891304443
        AR               674.827252   0.147541   2.744262 2018-01-21 22:18:06.593442627
        AT               646.650530   0.175066   2.872679 2018-01-21 15:54:42.469496094
        AU               669.558832   0.129808   2.843750 2018-01-22 02:28:39.199519287
        CA               648.747109   0.134534   2.951271 2018-01-22 14:40:47.165254150
        ...                     ...        ...        ...                           ...
        RU               662.994963   0.131258   2.832206 2018-01-21 07:11:16.534506104
        SE               660.612988   0.149020   2.682353 2018-01-22 07:48:23.447058838
        TR               485.253247   0.100000   1.900000 2018-01-16 16:02:33.000000000
        US               595.774391   0.125315   2.753900 2018-01-21 16:55:04.456970215
        ZA               643.053057   0.148410   2.766784 2018-01-22 15:17:56.141342773
        <BLANKLINE>
        [32 rows x 4 columns]
        """
        return self._query_compiler.aggs_groupby(
            by=self._by,
            pd_aggs=["mean"],
            dropna=self._dropna,
            numeric_only=numeric_only,
        )

    def var(self, numeric_only: bool = True) -> "pd.DataFrame":
        """
        Compute the variance value for each group.

        Parameters
        ----------
        numeric_only: {True, False, None} Default is True
            Which datatype to be returned
            - True: Returns all values as float64, NaN/NaT values are removed
            - None: Returns all values as the same dtype where possible, NaN/NaT are removed
            - False: Returns all values as the same dtype where possible, NaN/NaT are preserved

        Returns
        -------
        pandas.DataFrame
            variance value for each numeric column of each group

        See Also
        --------
        :pandas_api_docs:`pandas.core.groupby.GroupBy.var`

        Examples
        --------
        >>> df = ed.DataFrame(
        ...   "localhost", "flights",
        ...   columns=["AvgTicketPrice", "Cancelled", "dayOfWeek", "timestamp", "DestCountry"]
        ... )
        >>> df.groupby("DestCountry").var() # doctest: +NORMALIZE_WHITESPACE
                     AvgTicketPrice  Cancelled  dayOfWeek
        DestCountry
        AE             75789.979090   0.130443   3.950549
        AR             59683.055316   0.125979   3.783429
        AT             65726.669676   0.144610   4.090013
        AU             65088.483446   0.113094   3.833562
        CA             68149.950516   0.116496   3.688139
        ...                     ...        ...        ...
        RU             67305.277617   0.114107   3.852666
        SE             53740.570338   0.127062   3.942132
        TR             61245.521047   0.094868   4.100420
        US             74349.939410   0.109638   3.758700
        ZA             62920.072901   0.126608   3.775609
        <BLANKLINE>
        [32 rows x 3 columns]
        """
        return self._query_compiler.aggs_groupby(
            by=self._by,
            pd_aggs=["var"],
            dropna=self._dropna,
            numeric_only=numeric_only,
        )

    def std(self, numeric_only: bool = True) -> "pd.DataFrame":
        """
        Compute the standard deviation value for each group.

        Parameters
        ----------
        numeric_only: {True, False, None} Default is True
            Which datatype to be returned
            - True: Returns all values as float64, NaN/NaT values are removed
            - None: Returns all values as the same dtype where possible, NaN/NaT are removed
            - False: Returns all values as the same dtype where possible, NaN/NaT are preserved

        Returns
        -------
        pandas.DataFrame
            standard deviation value for each numeric column of each group

        See Also
        --------
        :pandas_api_docs:`pandas.core.groupby.GroupBy.std`

        Examples
        --------
        >>> df = ed.DataFrame(
        ...   "localhost", "flights",
        ...   columns=["AvgTicketPrice", "Cancelled", "dayOfWeek", "DestCountry"]
        ... )
        >>> df.groupby("DestCountry").std() # doctest: +NORMALIZE_WHITESPACE
                     AvgTicketPrice  Cancelled  dayOfWeek
        DestCountry
        AE               279.875500   0.367171   2.020634
        AR               244.903626   0.355811   1.949901
        AT               256.883342   0.381035   2.026411
        AU               255.585377   0.336902   1.961486
        CA               261.263054   0.341587   1.921980
        ...                     ...        ...        ...
        RU               259.696213   0.338140   1.964815
        SE               232.504297   0.357510   1.991340
        TR               267.827572   0.333333   2.191454
        US               272.774819   0.331242   1.939469
        ZA               251.505568   0.356766   1.948258
        <BLANKLINE>
        [32 rows x 3 columns]
        """
        return self._query_compiler.aggs_groupby(
            by=self._by,
            pd_aggs=["std"],
            dropna=self._dropna,
            numeric_only=numeric_only,
        )

    def mad(self, numeric_only: bool = True) -> "pd.DataFrame":
        """
        Compute the median absolute deviation value for each group.

        Parameters
        ----------
        numeric_only: {True, False, None} Default is True
            Which datatype to be returned
            - True: Returns all values as float64, NaN/NaT values are removed
            - None: Returns all values as the same dtype where possible, NaN/NaT are removed
            - False: Returns all values as the same dtype where possible, NaN/NaT are preserved

        Returns
        -------
        pandas.DataFrame
            median absolute deviation value for each numeric column of each group

        See Also
        --------
        :pandas_api_docs:`pandas.core.groupby.GroupBy.mad`

        Examples
        --------
        >>> df = ed.DataFrame(
        ...   "localhost", "flights",
        ...   columns=["AvgTicketPrice", "Cancelled", "dayOfWeek", "timestamp", "DestCountry"]
        ... )
        >>> df.groupby("DestCountry").mad() # doctest: +SKIP
                     AvgTicketPrice  Cancelled  dayOfWeek
        DestCountry
        AE               233.697174        NaN        1.5
        AR               189.250061        NaN        2.0
        AT               195.823669        NaN        2.0
        AU               202.539764        NaN        2.0
        CA               203.344696        NaN        2.0
        ...                     ...        ...        ...
        RU               206.431702        NaN        2.0
        SE               178.658447        NaN        2.0
        TR               221.863434        NaN        1.0
        US               228.461365        NaN        2.0
        ZA               192.162842        NaN        2.0
        <BLANKLINE>
        [32 rows x 3 columns]
        """
        return self._query_compiler.aggs_groupby(
            by=self._by,
            pd_aggs=["mad"],
            dropna=self._dropna,
            numeric_only=numeric_only,
        )

    def median(self, numeric_only: bool = True) -> "pd.DataFrame":
        """
        Compute the median value for each group.

        Parameters
        ----------
        numeric_only: {True, False, None} Default is True
            Which datatype to be returned
            - True: Returns all values as float64, NaN/NaT values are removed
            - None: Returns all values as the same dtype where possible, NaN/NaT are removed
            - False: Returns all values as the same dtype where possible, NaN/NaT are preserved

        Returns
        -------
        pandas.DataFrame
            median absolute deviation value for each numeric column of each group

        See Also
        --------
        :pandas_api_docs:`pandas.core.groupby.GroupBy.median`

        Examples
        --------
        >>> df = ed.DataFrame(
        ...   "localhost", "flights",
        ...   columns=["AvgTicketPrice", "Cancelled", "dayOfWeek", "timestamp", "DestCountry"]
        ... )
        >>> df.groupby("DestCountry").median(numeric_only=False) # doctest: +SKIP
                     AvgTicketPrice  Cancelled  dayOfWeek               timestamp
        DestCountry
        AE               585.720490      False          2 2018-01-19 23:56:44.000
        AR               678.447433      False          3 2018-01-22 10:18:50.000
        AT               659.715592      False          3 2018-01-20 20:40:10.000
        AU               689.241348      False          3 2018-01-22 18:46:11.000
        CA               663.516057      False          3 2018-01-22 21:35:09.500
        ...                     ...        ...        ...                     ...
        RU               670.714956      False          3 2018-01-20 16:48:16.000
        SE               680.111084      False          3 2018-01-22 20:53:44.000
        TR               441.681122      False          1 2018-01-13 23:17:27.000
        US               600.591525      False          3 2018-01-22 04:09:50.000
        ZA               633.935425      False          3 2018-01-23 17:42:57.000
        <BLANKLINE>
        [32 rows x 4 columns]
        """
        return self._query_compiler.aggs_groupby(
            by=self._by,
            pd_aggs=["median"],
            dropna=self._dropna,
            numeric_only=numeric_only,
        )

    def sum(self, numeric_only: bool = True) -> "pd.DataFrame":
        """
        Compute the sum value for each group.

        Parameters
        ----------
        numeric_only: {True, False, None} Default is True
            Which datatype to be returned
            - True: Returns all values as float64, NaN/NaT values are removed
            - None: Returns all values as the same dtype where possible, NaN/NaT are removed
            - False: Returns all values as the same dtype where possible, NaN/NaT are preserved

        Returns
        -------
        pandas.DataFrame
            sum value for each numeric column of each group

        See Also
        --------
        :pandas_api_docs:`pandas.core.groupby.GroupBy.sum`

        Examples
        --------
        >>> df = ed.DataFrame(
        ...   "localhost", "flights",
        ...   columns=["AvgTicketPrice", "Cancelled", "dayOfWeek", "DestCountry"]
        ... )
        >>> df.groupby("DestCountry").sum() # doctest: +NORMALIZE_WHITESPACE
                     AvgTicketPrice  Cancelled  dayOfWeek
        DestCountry
        AE             2.783612e+04        7.0      124.0
        AR             2.058223e+05       45.0      837.0
        AT             2.437872e+05       66.0     1083.0
        AU             2.785365e+05       54.0     1183.0
        CA             6.124173e+05      127.0     2786.0
        ...                     ...        ...        ...
        RU             4.899533e+05       97.0     2093.0
        SE             1.684563e+05       38.0      684.0
        TR             4.852532e+03        1.0       19.0
        US             1.183804e+06      249.0     5472.0
        ZA             1.819840e+05       42.0      783.0
        <BLANKLINE>
        [32 rows x 3 columns]
        """
        return self._query_compiler.aggs_groupby(
            by=self._by,
            pd_aggs=["sum"],
            dropna=self._dropna,
            numeric_only=numeric_only,
        )

    def min(self, numeric_only: bool = True) -> "pd.DataFrame":
        """
        Compute the min value for each group.

        Parameters
        ----------
        numeric_only: {True, False, None} Default is True
            Which datatype to be returned
            - True: Returns all values as float64, NaN/NaT values are removed
            - None: Returns all values as the same dtype where possible, NaN/NaT are removed
            - False: Returns all values as the same dtype where possible, NaN/NaT are preserved

        Returns
        -------
        pandas.DataFrame
            min value for each numeric column of each group

        See Also
        --------
        :pandas_api_docs:`pandas.core.groupby.GroupBy.min`

        Examples
        --------
        >>> df = ed.DataFrame(
        ...   "localhost", "flights",
        ...   columns=["AvgTicketPrice", "Cancelled", "dayOfWeek", "timestamp", "DestCountry"]
        ... )
        >>> df.groupby("DestCountry").min(numeric_only=False) # doctest: +NORMALIZE_WHITESPACE
                     AvgTicketPrice  Cancelled  dayOfWeek           timestamp
        DestCountry
        AE               110.799911      False          0 2018-01-01 19:31:30
        AR               125.589394      False          0 2018-01-01 01:30:47
        AT               100.020531      False          0 2018-01-01 05:24:19
        AU               102.294312      False          0 2018-01-01 00:00:00
        CA               100.557251      False          0 2018-01-01 00:44:08
        ...                     ...        ...        ...                 ...
        RU               101.004005      False          0 2018-01-01 01:01:51
        SE               102.877190      False          0 2018-01-01 04:09:38
        TR               142.876465      False          0 2018-01-01 06:45:17
        US               100.145966      False          0 2018-01-01 00:06:27
        ZA               102.002663      False          0 2018-01-01 06:44:44
        <BLANKLINE>
        [32 rows x 4 columns]
        """
        return self._query_compiler.aggs_groupby(
            by=self._by,
            pd_aggs=["min"],
            dropna=self._dropna,
            numeric_only=numeric_only,
        )

    def max(self, numeric_only: bool = True) -> "pd.DataFrame":
        """
        Compute the max value for each group.

        Parameters
        ----------
        numeric_only: {True, False, None} Default is True
            Which datatype to be returned
            - True: Returns all values as float64, NaN/NaT values are removed
            - None: Returns all values as the same dtype where possible, NaN/NaT are removed
            - False: Returns all values as the same dtype where possible, NaN/NaT are preserved

        Returns
        -------
        pandas.DataFrame
            max value for each numeric column of each group

        See Also
        --------
        :pandas_api_docs:`pandas.core.groupby.GroupBy.max`

        Examples
        --------
        >>> df = ed.DataFrame(
        ...   "localhost", "flights",
        ...   columns=["AvgTicketPrice", "Cancelled", "dayOfWeek", "timestamp", "DestCountry"]
        ... )
        >>> df.groupby("DestCountry").max(numeric_only=False) # doctest: +NORMALIZE_WHITESPACE
                     AvgTicketPrice  Cancelled  dayOfWeek           timestamp
        DestCountry
        AE              1126.148682       True          6 2018-02-11 04:11:14
        AR              1199.642822       True          6 2018-02-11 17:09:05
        AT              1181.835815       True          6 2018-02-11 23:12:33
        AU              1197.632690       True          6 2018-02-11 21:39:01
        CA              1198.852539       True          6 2018-02-11 23:04:08
        ...                     ...        ...        ...                 ...
        RU              1196.742310       True          6 2018-02-11 20:03:31
        SE              1198.621582       True          6 2018-02-11 22:06:14
        TR               855.935547       True          6 2018-02-04 01:59:23
        US              1199.729004       True          6 2018-02-11 23:27:00
        ZA              1196.186157       True          6 2018-02-11 23:29:45
        <BLANKLINE>
        [32 rows x 4 columns]
        """
        return self._query_compiler.aggs_groupby(
            by=self._by,
            pd_aggs=["max"],
            dropna=self._dropna,
            numeric_only=numeric_only,
        )

    def nunique(self) -> "pd.DataFrame":
        """
        Compute the nunique value for each group.

        Parameters
        ----------
        numeric_only: {True, False, None} Default is True
            Which datatype to be returned
            - True: Returns all values as float64, NaN/NaT values are removed
            - None: Returns all values as the same dtype where possible, NaN/NaT are removed
            - False: Returns all values as the same dtype where possible, NaN/NaT are preserved

        Returns
        -------
        pandas.DataFrame
            nunique value for each numeric column of each group

        See Also
        --------
        :pandas_api_docs:`pandas.core.groupby.GroupBy.nunique`

        Examples
        --------
        >>> df = ed.DataFrame(
        ...   "localhost", "flights",
        ...   columns=["AvgTicketPrice", "Cancelled", "dayOfWeek", "DestCountry"]
        ... )
        >>> df.groupby("DestCountry").nunique() # doctest: +NORMALIZE_WHITESPACE
                     AvgTicketPrice  Cancelled  dayOfWeek
        DestCountry
        AE                       46          2          7
        AR                      305          2          7
        AT                      377          2          7
        AU                      416          2          7
        CA                      944          2          7
        ...                     ...        ...        ...
        RU                      739          2          7
        SE                      255          2          7
        TR                       10          2          5
        US                     1987          2          7
        ZA                      283          2          7
        <BLANKLINE>
        [32 rows x 3 columns]
        """
        return self._query_compiler.aggs_groupby(
            by=self._by,
            pd_aggs=["nunique"],
            dropna=self._dropna,
            numeric_only=False,
        )

    def aggregate(
        self, func: Union[str, List[str]], numeric_only: Optional[bool] = False
    ) -> "pd.DataFrame":
        """
        Used to groupby and aggregate

        Parameters
        ----------
        func:
            Functions to use for aggregating the data.

            Accepted combinations are:
            - function
            - list of functions

        numeric_only: {True, False, None} Default is None
            Which datatype to be returned
            - True: returns all values with float64, NaN/NaT are ignored.
            - False: returns all values with float64.
            - None: returns all values with default datatype.

        Returns
        -------
        pandas.DataFrame
            aggregation value for each numeric column of each group

        See Also
        --------
        :pandas_api_docs:`pandas.core.groupby.GroupBy.aggregate`

        Examples
        --------
        >>> df = ed.DataFrame(
        ...   "localhost", "flights",
        ...   columns=["AvgTicketPrice", "Cancelled", "dayOfWeek", "DestCountry"]
        ... )
        >>> df.groupby("DestCountry").aggregate(["min", "max"]) # doctest: +NORMALIZE_WHITESPACE
                    AvgTicketPrice               ... dayOfWeek
                               min          max  ...       min max
        DestCountry                              ...
        AE              110.799911  1126.148682  ...         0   6
        AR              125.589394  1199.642822  ...         0   6
        AT              100.020531  1181.835815  ...         0   6
        AU              102.294312  1197.632690  ...         0   6
        CA              100.557251  1198.852539  ...         0   6
        ...                    ...          ...  ...       ...  ..
        RU              101.004005  1196.742310  ...         0   6
        SE              102.877190  1198.621582  ...         0   6
        TR              142.876465   855.935547  ...         0   6
        US              100.145966  1199.729004  ...         0   6
        ZA              102.002663  1196.186157  ...         0   6
        <BLANKLINE>
        [32 rows x 6 columns]
        """
        # Controls whether a MultiIndex is used for the
        # columns of the result DataFrame.
        is_dataframe_agg = True
        if isinstance(func, str):
            func = [func]
            is_dataframe_agg = False

        return self._query_compiler.aggs_groupby(
            by=self._by,
            pd_aggs=func,
            dropna=self._dropna,
            numeric_only=numeric_only,
            is_dataframe_agg=is_dataframe_agg,
        )

    agg = aggregate

    def count(self) -> "pd.DataFrame":
        """
        Compute the count value for each group.

        Returns
        -------
        pandas.DataFrame
            nunique value for each numeric column of each group

        See Also
        --------
        :pandas_api_docs:`pandas.core.groupby.GroupBy.count`

        Examples
        --------
        >>> df = ed.DataFrame(
        ...   "localhost", "flights",
        ...   columns=["AvgTicketPrice", "Cancelled", "dayOfWeek", "DestCountry"]
        ... )
        >>> df.groupby("DestCountry").count() # doctest: +NORMALIZE_WHITESPACE
                     AvgTicketPrice  Cancelled  dayOfWeek
        DestCountry
        AE                       46         46         46
        AR                      305        305        305
        AT                      377        377        377
        AU                      416        416        416
        CA                      944        944        944
        ...                     ...        ...        ...
        RU                      739        739        739
        SE                      255        255        255
        TR                       10         10         10
        US                     1987       1987       1987
        ZA                      283        283        283
        <BLANKLINE>
        [32 rows x 3 columns]
        """
        return self._query_compiler.aggs_groupby(
            by=self._by,
            pd_aggs=["count"],
            dropna=self._dropna,
            numeric_only=False,
            is_dataframe_agg=False,
        )

    def mode(self) -> None:
        raise NotImplementedError("Currently mode is not supported for groupby")
