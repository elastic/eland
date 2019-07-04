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

from modin.pandas.base import BasePandasDataset

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

    def _build_repr_df(self, num_rows, num_cols):
        # Overriden version of BasePandasDataset._build_repr_df
        # to avoid issues with concat
        if len(self.index) <= num_rows:
            return self.to_pandas()

        num_rows = num_rows + 1

        head_rows = int(num_rows / 2) + num_rows % 2
        tail_rows = num_rows - head_rows

        head = self.head(head_rows).to_pandas()
        tail = self.tail(tail_rows).to_pandas()

        return head.append(tail)

    def to_pandas(self):
        return self._query_compiler.to_pandas()
