import warnings
import sys

import pandas as pd
import numpy as np

from pandas.compat import StringIO
from pandas.core.common import apply_if_callable, is_bool_indexer
from pandas.io.common import _expand_user, _stringify_path
from pandas.io.formats import console
from pandas.io.formats import format as fmt
from pandas.io.formats.printing import pprint_thing

from eland import NDFrame
from eland import Series


class DataFrame(NDFrame):
    # TODO create effectively 2 constructors
    # 1. client, index_pattern, columns, index_field
    # 2. query_compiler
    def __init__(self,
                 client=None,
                 index_pattern=None,
                 columns=None,
                 index_field=None,
                 query_compiler=None):
        # python 3 syntax
        super().__init__(
            client=client,
            index_pattern=index_pattern,
            columns=columns,
            index_field=index_field,
            query_compiler=query_compiler)

    def _get_columns(self):
        return self._query_compiler.columns

    columns = property(_get_columns)

    @property
    def empty(self):
        """Determines if the DataFrame is empty.

        Returns:
            True if the DataFrame is empty.
            False otherwise.
        """
        # TODO - this is called on every attribute get (most methods) from modin/pandas/base.py:3337
        #  (as Index.__len__ performs an query) we may want to cache self.index.empty()
        return len(self.columns) == 0 or len(self.index) == 0

    def head(self, n=5):
        return super().head(n)

    def tail(self, n=5):
        return super().tail(n)

    def __repr__(self):
        """
        From pandas
        """
        buf = StringIO()

        max_rows = pd.get_option("display.max_rows")
        max_cols = pd.get_option("display.max_columns")
        show_dimensions = pd.get_option("display.show_dimensions")
        if pd.get_option("display.expand_frame_repr"):
            width, _ = console.get_console_size()
        else:
            width = None
        self.to_string(buf=buf, max_rows=max_rows, max_cols=max_cols,
                       line_width=width, show_dimensions=show_dimensions)

        return buf.getvalue()

    def info_es(self):
        buf = StringIO()

        super().info_es(buf)

        return buf.getvalue()



    def to_string(self, buf=None, columns=None, col_space=None, header=True,
                  index=True, na_rep='NaN', formatters=None, float_format=None,
                  sparsify=None, index_names=True, justify=None,
                  max_rows=None, max_cols=None, show_dimensions=False,
                  decimal='.', line_width=None):
        """
        From pandas - except we set max_rows default to avoid careless extraction of entire index
        """
        if max_rows is None:
            warnings.warn("DataFrame.to_string called without max_rows set "
                          "- this will return entire index results. "
                          "Setting max_rows=60, overwrite if different behaviour is required.")
            max_rows = 60

        # Create a slightly bigger dataframe than display
        df = self._build_repr_df(max_rows+1, max_cols)

        if buf is not None:
            _buf = _expand_user(_stringify_path(buf))
        else:
            _buf = StringIO()

        df.to_string(buf=_buf, columns=columns,
                     col_space=col_space, na_rep=na_rep,
                     formatters=formatters,
                     float_format=float_format,
                     sparsify=sparsify, justify=justify,
                     index_names=index_names,
                     header=header, index=index,
                     max_rows=max_rows,
                     max_cols=max_cols,
                     show_dimensions=False,  # print this outside of this call
                     decimal=decimal,
                     line_width=line_width)

        # Our fake dataframe has incorrect number of rows (max_rows*2+1) - write out
        # the correct number of rows
        if show_dimensions:
            _buf.write("\n\n[{nrows} rows x {ncols} columns]"
                       .format(nrows=len(self.index), ncols=len(self.columns)))

        if buf is None:
            result = _buf.getvalue()
            return result

    def _getitem(self, key):
        """Get the column specified by key for this DataFrame.

        Args:
            key : The column name.

        Returns:
            A Pandas Series representing the value for the column.
        """
        key = apply_if_callable(key, self)
        # Shortcut if key is an actual column
        try:
            if key in self.columns:
                return self._getitem_column(key)
        except (KeyError, ValueError, TypeError):
            pass
        if isinstance(key, (Series, np.ndarray, pd.Index, list)):
            return self._getitem_array(key)
        elif isinstance(key, DataFrame):
            return self.where(key)
        else:
            return self._getitem_column(key)

    def _getitem_column(self, key):
        if key not in self.columns:
            raise KeyError("{}".format(key))
        s = self._reduce_dimension(self._query_compiler.getitem_column_array([key]))
        s._parent = self
        return s

    def _getitem_array(self, key):
        if isinstance(key, Series):
            key = key._to_pandas()
        if is_bool_indexer(key):
            if isinstance(key, pd.Series) and not key.index.equals(self.index):
                warnings.warn(
                    "Boolean Series key will be reindexed to match DataFrame index.",
                    PendingDeprecationWarning,
                    stacklevel=3,
                )
            elif len(key) != len(self.index):
                raise ValueError(
                    "Item wrong length {} instead of {}.".format(
                        len(key), len(self.index)
                    )
                )
            key = check_bool_indexer(self.index, key)
            # We convert to a RangeIndex because getitem_row_array is expecting a list
            # of indices, and RangeIndex will give us the exact indices of each boolean
            # requested.
            key = pd.RangeIndex(len(self.index))[key]
            if len(key):
                return DataFrame(
                    query_compiler=self._query_compiler.getitem_row_array(key)
                )
            else:
                return DataFrame(columns=self.columns)
        else:
            if any(k not in self.columns for k in key):
                raise KeyError(
                    "{} not index".format(
                        str([k for k in key if k not in self.columns]).replace(",", "")
                    )
                )
            return DataFrame(
                query_compiler=self._query_compiler.getitem_column_array(key)
            )

    def _reduce_dimension(self, query_compiler):
        return Series(query_compiler=query_compiler)

    def _to_pandas(self):
        return self._query_compiler.to_pandas()

    def squeeze(self, axis=None):
        return DataFrame(
            query_compiler=self._query_compiler.squeeze(axis)
        )
