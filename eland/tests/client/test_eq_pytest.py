# File called _pytest for PyCharm compatability
from elasticsearch import Elasticsearch

import eland as ed
from eland.tests.common import TestData


class TestClientEq(TestData):

    def test_self_eq(self):
        es = Elasticsearch('localhost')

        client = ed.Client(es)

        assert client != es

        assert client == client

    def test_non_self_ne(self):
        es1 = Elasticsearch('localhost')
        es2 = Elasticsearch('localhost')

        client1 = ed.Client(es1)
        client2 = ed.Client(es2)

        assert client1 != client2
