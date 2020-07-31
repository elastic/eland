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

from typing import Optional, TextIO, TYPE_CHECKING, Union, Sequence, Tuple, List
from eland.utils import deprecated_api

if TYPE_CHECKING:
    from .query_compiler import QueryCompiler


def make_index(
    query_compiler: "QueryCompiler",
    es_index_fields: Optional[Union[str, Sequence[str]]],
) -> "Index":
    if es_index_fields is None:
        es_index_fields = (Index.ID_INDEX_FIELD,)
    elif isinstance(es_index_fields, str):
        es_index_fields = (es_index_fields,)
    if len(es_index_fields) == 1:
        return Index(query_compiler, es_index_fields=es_index_fields[0])
    else:
        return MultiIndex(query_compiler, es_index_fields=es_index_fields)


class Index:
    """
    The index for an eland.DataFrame.

    TODO - This currently has very different behaviour than pandas.Index

    Currently, the index is a field that exists in every document in an Elasticsearch index.
    For slicing and sorting operations it must be a docvalues field. By default _id is used,
    which can't be used for range queries and is inefficient for sorting:

    `<https://www.elastic.co/guide/en/elasticsearch/reference/current/mapping-id-field.html>`_

    (The value of the _id field is also accessible in aggregations or for sorting,
    but doing so is discouraged as it requires to load a lot of data in memory.
    In case sorting or aggregating on the _id field is required, it is advised to duplicate
    the content of the _id field in another field that has doc_values enabled.)
    """

    ID_INDEX_FIELD = "_id"
    ID_SORT_FIELD = "_doc"  # if index field is _id, sort by _doc

    def __init__(
        self,
        query_compiler: "QueryCompiler",
        es_index_fields: Optional[Union[str, Sequence[str]]] = None,
    ):
        self._query_compiler = query_compiler

        # _is_source_field is set immediately within
        # index_field.setter
        self._is_source_field = False

        # The type:ignore is due to mypy not being smart enough
        # to recognize the property.setter has a different type
        # than the property.getter.
        self.es_index_fields = es_index_fields  # type: ignore

    def __len__(self) -> int:
        return self._query_compiler._index_count()

    # Make iterable
    def __next__(self) -> None:
        # TODO resolve this hack to make this 'iterable'
        raise StopIteration()

    def __iter__(self) -> "Index":
        return self

    @property
    def shape(self) -> Tuple[int]:
        """Return a tuple of the shape of the underlying data."""
        return (len(self),)

    @property
    def size(self) -> int:
        """Return the number of elements in the underlying data."""
        return len(self)

    def es_info(self, buf: TextIO) -> None:
        buf.write(f"{type(self).__name__}:\n")
        buf.write(f" es_index_fields: {self.es_index_fields}\n")
        buf.write(f" is_source_fields: {self.is_source_fields}\n")

    @deprecated_api("eland.Index.es_info()")
    def info_es(self, buf: TextIO) -> None:
        self.es_info(buf)

    @property
    def es_index_fields(self) -> Tuple[str, ...]:
        return self._es_index_fields

    @es_index_fields.setter
    def es_index_fields(self, value: Optional[Union[str, Sequence[str]]]) -> None:
        if value is None:
            value = (Index.ID_INDEX_FIELD,)
        elif isinstance(value, str):
            value = (value,)
        self._es_index_fields = tuple(value)

    @property
    def sort_fields(self) -> Tuple[str, ...]:
        return tuple(
            [
                es_field if es_field != Index.ID_INDEX_FIELD else Index.ID_SORT_FIELD
                for es_field in self.es_index_fields
            ]
        )

    @property
    def is_source_fields(self) -> Tuple[Tuple[str, bool], ...]:
        return tuple(
            [
                (es_field, es_field != Index.ID_INDEX_FIELD)
                for es_field in self.es_index_fields
            ]
        )

    @property
    def names(self) -> List[Optional[str]]:
        """Names of levels in MultiIndex."""
        return [None]

    @property
    def name(self) -> Optional[str]:
        """Return Index or MultiIndex name."""
        return None


class MultiIndex(Index):
    """A multi-level index of an eland.DataFrame. As is the case for
    a single-level eland.Index this index has different behavior than
    a pandas.Index.

    To use a MultiIndex you can construct your DataFrame by
    setting the ``es_index_field`` argument to a list or tuple
    of strings:

        >>> df = ed.DataFrame(
        ...     es_client='localhost',
        ...     es_index_pattern='flights',
        ...     es_index_field=["OriginCountry", "DestCountry"],
        ... )
                                   AvgTicketPrice  ...           timestamp
        OriginCountry DestCountry                  ...
        AE            AE               441.242460  ... 2018-01-06 13:03:25
                      AE               190.500200  ... 2018-01-08 05:25:09
                      AE               110.799908  ... 2018-01-09 10:03:42
                      AE               185.232228  ... 2018-01-30 22:16:18
                      AE               212.574590  ... 2018-02-05 16:48:34
        ...                                   ...  ...                 ...
                      AE               441.242460  ... 2018-01-06 13:03:25
                      AE               190.500200  ... 2018-01-08 05:25:09
                      AE               110.799908  ... 2018-01-09 10:03:42
                      AE               185.232228  ... 2018-01-30 22:16:18
                      AE               212.574590  ... 2018-02-05 16:48:34
        <BLANKLINE>
        [13059 rows x 27 columns]
    """

    @property
    def names(self) -> List[Optional[str]]:
        """Names of levels in MultiIndex."""
        return list(self.es_index_fields)
