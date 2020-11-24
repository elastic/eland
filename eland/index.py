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

from typing import TYPE_CHECKING, Optional, TextIO

if TYPE_CHECKING:
    from .query_compiler import QueryCompiler


class Index:
    """
    The index for an eland.DataFrame.

    TODO - This currently has very different behaviour than pandas.Index

    Currently, the index is a field that exists in every document in an Elasticsearch index.
    For slicing and sorting operations it must be a docvalues field. By default _id is used,
    which can't be used for range queries and is inefficient for sorting:

    https://www.elastic.co/guide/en/elasticsearch/reference/current/mapping-id-field.html
    (The value of the _id field is also accessible in aggregations or for sorting,
    but doing so is discouraged as it requires to load a lot of data in memory.
    In case sorting or aggregating on the _id field is required, it is advised to duplicate
    the content of the _id field in another field that has doc_values enabled.)
    """

    ID_INDEX_FIELD = "_id"
    ID_SORT_FIELD = "_doc"  # if index field is _id, sort by _doc

    def __init__(
        self, query_compiler: "QueryCompiler", es_index_field: Optional[str] = None
    ):
        self._query_compiler = query_compiler

        # _is_source_field is set immediately within
        # index_field.setter
        self._is_source_field = False

        # The type:ignore is due to mypy not being smart enough
        # to recognize the property.setter has a different type
        # than the property.getter.
        self.es_index_field = es_index_field  # type: ignore

    @property
    def sort_field(self) -> str:
        if self._index_field == self.ID_INDEX_FIELD:
            return self.ID_SORT_FIELD
        return self._index_field

    @property
    def is_source_field(self) -> bool:
        return self._is_source_field

    @property
    def es_index_field(self) -> str:
        return self._index_field

    @es_index_field.setter
    def es_index_field(self, index_field: Optional[str]) -> None:
        if index_field is None or index_field == Index.ID_INDEX_FIELD:
            self._index_field = Index.ID_INDEX_FIELD
            self._is_source_field = False
        else:
            self._index_field = index_field
            self._is_source_field = True

    def __len__(self) -> int:
        return self._query_compiler._index_count()

    # Make iterable
    def __next__(self) -> None:
        # TODO resolve this hack to make this 'iterable'
        raise StopIteration()

    def __iter__(self) -> "Index":
        return self

    def es_info(self, buf: TextIO) -> None:
        buf.write("Index:\n")
        buf.write(f" es_index_field: {self.es_index_field}\n")
        buf.write(f" is_source_field: {self.is_source_field}\n")
