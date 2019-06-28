"""
class Index

The index for an eland.DataFrame.

Currently, the index is a field that exists in every document in an Elasticsearch index.
For slicing and sorting operations it must be a docvalues field. By default _id is used,
which can't be used for range queries and is inefficient for sorting:

https://www.elastic.co/guide/en/elasticsearch/reference/current/mapping-id-field.html
(The value of the _id field is also accessible in aggregations or for sorting,
but doing so is discouraged as it requires to load a lot of data in memory.
In case sorting or aggregating on the _id field is required, it is advised to duplicate
the content of the _id field in another field that has doc_values enabled.)

"""
class Index:
    ID_INDEX_FIELD = '_id'
    ID_SORT_FIELD = '_doc' # if index field is _id, sort by _doc

    def __init__(self, index_field=None):
        # Calls setter
        self.index_field = index_field

    @property
    def sort_field(self):
        if self._index_field == self.ID_INDEX_FIELD:
            return self.ID_SORT_FIELD
        return self._index_field

    @property
    def is_source_field(self):
        return self._is_source_field

    @property
    def index_field(self):
        return self._index_field

    @index_field.setter
    def index_field(self, index_field):
        if index_field == None:
            self._index_field = Index.ID_INDEX_FIELD
            self._is_source_field = False
        else:
            self._index_field = index_field
            self._is_source_field = True
