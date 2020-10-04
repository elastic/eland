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

from abc import ABC
from typing import List, Optional, Union
from eland.query_compiler import QueryCompiler


class GroupBy(ABC):
    """
    This holds all the groupby base methods

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
        by: Union[str, List[str], None] = None,
        query_compiler: Optional["QueryCompiler"] = None,
        dropna: bool = True,
    ) -> None:
        self._query_compiler: "QueryCompiler" = QueryCompiler(to_copy=query_compiler)
        self._dropna: bool = dropna
        self._by: Union[str, List[str]] = by

    def mean(self, numeric_only: bool = True):
        # numeric_only=True becuase pandas does the same
        return self._query_compiler.groupby(
            by=self._by,
            pd_aggs=["mean"],
            dropna=self._dropna,
            numeric_only=numeric_only,
        )

    def var(self, numeric_only: bool = True):
        # numeric_only=True becuase pandas does the same
        return self._query_compiler.groupby(
            by=self._by,
            pd_aggs=["var"],
            dropna=self._dropna,
            numeric_only=numeric_only,
        )

    def std(self, numeric_only: bool = True):
        # numeric_only=True becuase pandas does the same
        return self._query_compiler.groupby(
            by=self._by,
            pd_aggs=["std"],
            dropna=self._dropna,
            numeric_only=numeric_only,
        )

    def mad(self, numeric_only: bool = True):
        # numeric_only=True becuase pandas does the same
        return self._query_compiler.groupby(
            by=self._by,
            pd_aggs=["mad"],
            dropna=self._dropna,
            numeric_only=numeric_only,
        )

    def median(self, numeric_only: bool = True):
        # numeric_only=True becuase pandas does the same
        return self._query_compiler.groupby(
            by=self._by,
            pd_aggs=["median"],
            dropna=self._dropna,
            numeric_only=numeric_only,
        )

    def sum(self, numeric_only: bool = True):
        # numeric_only=True becuase pandas does the same
        return self._query_compiler.groupby(
            by=self._by,
            pd_aggs=["sum"],
            dropna=self._dropna,
            numeric_only=numeric_only,
        )

    def min(self, numeric_only: bool = True):
        # numeric_only=True becuase pandas does the same
        return self._query_compiler.groupby(
            by=self._by,
            pd_aggs=["min"],
            dropna=self._dropna,
            numeric_only=numeric_only,
        )

    def max(self, numeric_only: bool = True):
        # numeric_only=True becuase pandas does the same
        return self._query_compiler.groupby(
            by=self._by,
            pd_aggs=["max"],
            dropna=self._dropna,
            numeric_only=numeric_only,
        )

    def nunique(self):
        return self._query_compiler.groupby(
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

    def __init__(
        self,
        by: Union[str, List[str], None] = None,
        query_compiler: Optional["QueryCompiler"] = None,
        dropna: bool = True,
    ) -> None:
        super().__init__(by=by, query_compiler=query_compiler, dropna=dropna)

    def aggregate(self, func: Union[str, List[str]], numeric_only: bool = False):
        """
        Used to groupby and aggregate

        Parameters
        ----------
        func:
            Functions to use for aggregating the data.

            Accepted combinations are:
            - function
            - list of functions
            TODO Implement other functions present in pandas groupby
        numeric_only: {True, False, None} Default is None
            Which datatype to be returned
            - True: returns all values with float64, NaN/NaT are ignored.
            - False: returns all values with float64.
            - None: returns all values with default datatype.
        """
        # numeric_only is by default False because pandas does the same
        return self._query_compiler.groupby(
            by=self._by,
            pd_aggs=([func] if isinstance(func, str) else func),
            dropna=self._dropna,
            numeric_only=numeric_only,
            is_agg=True,
        )

    agg = aggregate
