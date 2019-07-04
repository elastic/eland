import warnings

import pandas as pd
from pandas.compat import StringIO
from pandas.io.common import _expand_user, _stringify_path
from pandas.io.formats import console

from eland import NDFrame


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

    def to_string(self, buf=None, columns=None, col_space=None, header=True,
                  index=True, na_rep='NaN', formatters=None, float_format=None,
                  sparsify=None, index_names=True, justify=None,
                  max_rows=None, max_cols=None, show_dimensions=False,
                  decimal='.', line_width=None):
        """
        From pandas - except we set max_rows default to avoid careless
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
