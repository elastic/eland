from eland import NDFrame

import pandas as pd


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
        num_rows = pd.get_option("max_rows") or 60
        num_cols = pd.get_option("max_columns") or 20

        result = repr(self._build_repr_df(num_rows, num_cols))
        if len(self.index) > num_rows or len(self.columns) > num_cols:
            # The split here is so that we don't repr pandas row lengths.
            return result.rsplit("\n\n", 1)[0] + "\n\n[{0} rows x {1} columns]".format(
                len(self.index), len(self.columns)
            )
        else:
            return result
