"""
GroupBy
---------
Define the SeriesGroupBy, DataFrameGroupBy, and PanelGroupBy
classes that hold the groupby interfaces (and some implementations).

These are user facing as the result of the ``df.groupby(...)`` operations,
which here returns a DataFrameGroupBy object.
"""

#  Copyright 2019 Elasticsearch BV
#
#      Licensed under the Apache License, Version 2.0 (the "License");
#      you may not use this file except in compliance with the License.
#      You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#      Unless required by applicable law or agreed to in writing, software
#      distributed under the License is distributed on an "AS IS" BASIS,
#      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#      See the License for the specific language governing permissions and
#      limitations under the License.

from eland import NDFrame


class DataFrameGroupBy(NDFrame):

    def __init__(self,
                 df,
                 by):
        super().__init__(
            query_compiler=df._query_compiler.copy()
        )
        self._query_compiler.groupby_agg(by)
