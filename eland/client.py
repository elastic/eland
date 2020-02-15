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

from elasticsearch import Elasticsearch
from elasticsearch import helpers


class Client:
    """
    eland client - implemented as facade to control access to Elasticsearch methods
    """

    def __init__(self, es=None):
        if isinstance(es, Elasticsearch):
            self._es = es
        elif isinstance(es, Client):
            self._es = es._es
        else:
            self._es = Elasticsearch(es)

    def index_create(self, **kwargs):
        return self._es.indices.create(**kwargs)

    def index_delete(self, **kwargs):
        return self._es.indices.delete(**kwargs)

    def index_exists(self, **kwargs):
        return self._es.indices.exists(**kwargs)

    def get_mapping(self, **kwargs):
        return self._es.indices.get_mapping(**kwargs)

    def bulk(self, actions, refresh=False):
        return helpers.bulk(self._es, actions, refresh=refresh)

    def scan(self, **kwargs):
        return helpers.scan(self._es, **kwargs)

    def search(self, **kwargs):
        return self._es.search(**kwargs)

    def field_caps(self, **kwargs):
        return self._es.field_caps(**kwargs)

    def count(self, **kwargs):
        count_json = self._es.count(**kwargs)
        return count_json['count']

    def perform_request(self, method, url, headers=None, params=None, body=None):
        return self._es.transport.perform_request(method, url, headers, params, body)
