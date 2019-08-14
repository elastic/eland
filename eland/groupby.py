"""
GroupBy
---------
Define the SeriesGroupBy, DataFrameGroupBy, and PanelGroupBy
classes that hold the groupby interfaces (and some implementations).

These are user facing as the result of the ``df.groupby(...)`` operations,
which here returns a DataFrameGroupBy object.
"""

from eland import NDFrame


class DataFrameGroupBy(NDFrame):

    def __init__(self,
                 df,
                 by):
        super().__init__(
            query_compiler=df._query_compiler.copy()
        )
        self._query_compiler.groupby_agg(by)
