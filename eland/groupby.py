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

from typing import TYPE_CHECKING, List

from eland.query_compiler import QueryCompiler

if TYPE_CHECKING:
    import pandas as pd  # type: ignore


class GroupBy:
    """
    Base class for calls to X.groupby([...])

    Parameters
    ----------
    by:
        List of columns to groupby
    query_compiler:
        Query compiler object
    dropna:
        default is true, drop None/NaT/NaN values while grouping
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

    def mean(self, numeric_only: bool = True) -> "pd.DataFrame":
        return self._query_compiler.aggs_groupby(
            by=self._by,
            pd_aggs=["mean"],
            dropna=self._dropna,
            numeric_only=numeric_only,
        )

    def var(self, numeric_only: bool = True) -> "pd.DataFrame":
        return self._query_compiler.aggs_groupby(
            by=self._by,
            pd_aggs=["var"],
            dropna=self._dropna,
            numeric_only=numeric_only,
        )

    def std(self, numeric_only: bool = True) -> "pd.DataFrame":
        return self._query_compiler.aggs_groupby(
            by=self._by,
            pd_aggs=["std"],
            dropna=self._dropna,
            numeric_only=numeric_only,
        )

    def mad(self, numeric_only: bool = True) -> "pd.DataFrame":
        return self._query_compiler.aggs_groupby(
            by=self._by,
            pd_aggs=["mad"],
            dropna=self._dropna,
            numeric_only=numeric_only,
        )

    def median(self, numeric_only: bool = True) -> "pd.DataFrame":
        return self._query_compiler.aggs_groupby(
            by=self._by,
            pd_aggs=["median"],
            dropna=self._dropna,
            numeric_only=numeric_only,
        )

    def sum(self, numeric_only: bool = True) -> "pd.DataFrame":
        return self._query_compiler.aggs_groupby(
            by=self._by,
            pd_aggs=["sum"],
            dropna=self._dropna,
            numeric_only=numeric_only,
        )

    def min(self, numeric_only: bool = True) -> "pd.DataFrame":
        return self._query_compiler.aggs_groupby(
            by=self._by,
            pd_aggs=["min"],
            dropna=self._dropna,
            numeric_only=numeric_only,
        )

    def max(self, numeric_only: bool = True) -> "pd.DataFrame":
        return self._query_compiler.aggs_groupby(
            by=self._by,
            pd_aggs=["max"],
            dropna=self._dropna,
            numeric_only=numeric_only,
        )

    def nunique(self) -> "pd.DataFrame":
        return self._query_compiler.aggs_groupby(
            by=self._by,
            pd_aggs=["nunique"],
            dropna=self._dropna,
            numeric_only=False,
        )


class GroupByDataFrame(GroupBy):
    """
    This holds all the groupby methods for DataFrame

    Parameters
    ----------
    by:
        List of columns to groupby
    query_compiler:
        Query compiler object
    dropna:
        default is true, drop None/NaT/NaN values while grouping
    """

    def aggregate(self, func: List[str], numeric_only: bool = False) -> "pd.DataFrame":
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
        """
        if isinstance(func, str):
            func = [func]
        return self._query_compiler.aggs_groupby(
            by=self._by,
            pd_aggs=func,
            dropna=self._dropna,
            numeric_only=numeric_only,
            is_dataframe_agg=True,
        )

    agg = aggregate
