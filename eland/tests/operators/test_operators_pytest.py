# -*- coding: UTF-8 -*-
from eland.filter import *


class TestOperators:
    def test_leaf_boolean_filter(self):
        assert GreaterEqual('a', 2).build() == {"range": {"a": {"gte": 2}}}
        assert LessEqual('a', 2).build() == {"range": {"a": {"lte": 2}}}
        assert Less('a', 2).build() == {"range": {"a": {"lt": 2}}}
        assert Equal('a', 2).build() == {"term": {"a": 2}}
        exp = Equal('a', 2)
        assert (~exp).build()['bool'], {"must_not": {"term": {"a": 2}}}
        assert Greater('a', 2).build() == {"range": {"a": {"gt": 2}}}
        assert IsIn('a', [1, 2, 3]).build() == {'terms': {'a': [1, 2, 3]}}
        assert Like('a', 'a*b').build() == {'wildcard': {'a': 'a*b'}}
        assert Rlike('a', 'a*b').build() == {'regexp': {'a': 'a*b'}}
        assert Startswith('a', 'jj').build() == {'prefix': {'a': 'jj'}}
        assert IsNull('a').build() == {'missing': {'field': 'a'}}
        assert NotNull('a').build() == {'exists': {'field': 'a'}}
        assert ScriptFilter('doc["num1"].value > params.param1', params={'param1': 5}).build() == {
            'script': {'script': {'inline': 'doc["num1"].value > params.param1', 'params': {'param1': 5}}}}
        assert IsIn('ids', [1, 2, 3]).build() == {'ids': {'values': [1, 2, 3]}}

    def test_and_filter1(self):
        exp = GreaterEqual('a', 2) & Less('b', 3)
        assert exp.build() == {'bool': {'must': [{'range': {'a': {'gte': 2}}}, {'range': {'b': {'lt': 3}}}]}}

    def test_and_filter2(self):
        exp = GreaterEqual('a', 2) & Less('b', 3) & Equal('c', 4)
        assert exp.build() == \
               {
                   'bool': {
                       'must': [
                           {'range': {'a': {'gte': 2}}},
                           {'range': {'b': {'lt': 3}}},
                           {'term': {'c': 4}}
                       ]
                   }
               }

    def test_and_filter3(self):
        exp = GreaterEqual('a', 2) & (Less('b', 3) & Equal('c', 4))
        assert exp.build() == \
               {
                   'bool': {
                       'must': [
                           {'range': {'b': {'lt': 3}}},
                           {'term': {'c': 4}},
                           {'range': {'a': {'gte': 2}}}
                       ]
                   }
               }

    def test_or_filter1(self):
        exp = GreaterEqual('a', 2) | Less('b', 3)
        assert exp.build() == \
               {
                   'bool': {
                       'should': [
                           {'range': {'a': {'gte': 2}}},
                           {'range': {'b': {'lt': 3}}}
                       ]
                   }
               }

    def test_or_filter2(self):
        exp = GreaterEqual('a', 2) | Less('b', 3) | Equal('c', 4)
        assert exp.build() == \
               {
                   'bool': {
                       'should': [
                           {'range': {'a': {'gte': 2}}},
                           {'range': {'b': {'lt': 3}}},
                           {'term': {'c': 4}}
                       ]
                   }
               }

    def test_or_filter3(self):
        exp = GreaterEqual('a', 2) | (Less('b', 3) | Equal('c', 4))
        assert exp.build() == \
               {
                   'bool': {
                       'should': [
                           {'range': {'b': {'lt': 3}}},
                           {'term': {'c': 4}},
                           {'range': {'a': {'gte': 2}}}
                       ]
                   }
               }

    def test_not_filter(self):
        exp = ~GreaterEqual('a', 2)
        assert exp.build() == \
               {
                   'bool': {
                       'must_not': {'range': {'a': {'gte': 2}}}
                   }
               }

    def test_not_not_filter(self):
        exp = ~~GreaterEqual('a', 2)

        assert exp.build() == \
               {
                   'bool': {
                       'must_not': {
                           'bool': {
                               'must_not': {'range': {'a': {'gte': 2}}}
                           }
                       }
                   }
               }

    def test_not_and_filter(self):
        exp = ~(GreaterEqual('a', 2) & Less('b', 3))
        assert exp.build() == \
               {
                   'bool': {
                       'must_not': {
                           'bool': {
                               'must': [
                                   {'range': {'a': {'gte': 2}}},
                                   {'range': {'b': {'lt': 3}}}
                               ]
                           }
                       }
                   }
               }

    def test_and_or_filter(self):
        exp = GreaterEqual('a', 2) & (Less('b', 3) | Equal('c', 4))
        assert exp.build() == \
               {
                   'bool': {
                       'must': [
                           {'range': {'a': {'gte': 2}}},
                           {
                               'bool': {
                                   'should': [
                                       {'range': {'b': {'lt': 3}}},
                                       {'term': {'c': 4}}
                                   ]
                               }
                           }
                       ]
                   }
               }

    def test_and_not_or_filter(self):
        exp = GreaterEqual('a', 2) & ~(Less('b', 3) | Equal('c', 4))
        assert exp.build() == \
               {
                   'bool': {
                       'must': [
                           {'range': {'a': {'gte': 2}}},
                           {
                               'bool': {
                                   'must_not': {
                                       'bool': {
                                           'should': [
                                               {'range': {'b': {'lt': 3}}},
                                               {'term': {'c': 4}}
                                           ]
                                       }

                                   }
                               }
                           }
                       ]
                   }
               }
