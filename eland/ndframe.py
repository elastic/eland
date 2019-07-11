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
from modin.pandas.base import BasePandasDataset
from modin.pandas.indexing import _iLocIndexer
from pandas.util._validators import validate_bool_kwarg
from pandas.core.dtypes.common import is_list_like

from eland import ElandQueryCompiler


class NDFrame(BasePandasDataset):

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
            query_compiler = ElandQueryCompiler(client=client,
                                                index_pattern=index_pattern,
                                                columns=columns,
                                                index_field=index_field)
        self._query_compiler = query_compiler

    def _get_index(self):
        return self._query_compiler.index

    index = property(_get_index)

    @property
    def dtypes(self):
        return self._query_compiler.dtypes

    def get_dtype_counts(self):
        return self._query_compiler.get_dtype_counts()

    def _build_repr_df(self, num_rows, num_cols):
        # Overriden version of BasePandasDataset._build_repr_df
        # to avoid issues with concat
        if len(self.index) <= num_rows:
            return self._to_pandas()

        num_rows = num_rows

        head_rows = int(num_rows / 2) + num_rows % 2
        tail_rows = num_rows - head_rows

        head = self.head(head_rows)._to_pandas()
        tail = self.tail(tail_rows)._to_pandas()

        return head.append(tail)

    def __getattr__(self, key):
        """After regular attribute access, looks up the name in the columns

        Args:
            key (str): Attribute name.

        Returns:
            The value of the attribute.
        """
        print(key)
        try:
            return object.__getattribute__(self, key)
        except AttributeError as e:
            if key in self.columns:
                return self[key]
            raise e

    def __sizeof__(self):
        # Don't default to pandas, just return approximation TODO - make this more accurate
        return sys.getsizeof(self._query_compiler)

    @property
    def iloc(self):
        """Purely integer-location based indexing for selection by position.

        """
        return _iLocIndexer(self)

    def info_es(self, buf):
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
        Args:
            labels: Index or column labels to drop.
            axis: Whether to drop labels from the index (0 / 'index') or
                columns (1 / 'columns').
            index, columns: Alternative to specifying axis (labels, axis=1 is
                equivalent to columns=labels).
            level: For MultiIndex
            inplace: If True, do operation inplace and return None.
            errors: If 'ignore', suppress error and existing labels are
                dropped.
        Returns:
            dropped : type of caller

        (derived from modin.base.BasePandasDataset)
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

    # TODO implement arguments
    def mean(self):
        return self._query_compiler.mean()

    def sum(self, numeric_only=True):
        if numeric_only == False:
            raise NotImplementedError("Only sum of numeric fields is implemented")
        return self._query_compiler.sum()

    def min(self, numeric_only=True):
        if numeric_only == False:
            raise NotImplementedError("Only sum of numeric fields is implemented")
        return self._query_compiler.min()

    def max(self, numeric_only=True):
        if numeric_only == False:
            raise NotImplementedError("Only sum of numeric fields is implemented")
        return self._query_compiler.max()

    def describe(self):
        return self._query_compiler.describe()
