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

# Originally based on code in pandasticsearch filters

# Es filter builder for BooleanCond
class BooleanFilter:
    def __init__(self, *args):
        self._filter = None

    def __and__(self, x):
        # Combine results
        if isinstance(self, AndFilter):
            if 'must_not' in x.subtree:
                # nest a must_not under a must
                self.subtree['must'].append(x.build())  # 'build includes bool'
            else:
                # append a must to a must
                self.subtree['must'].append(x.subtree)  # 'subtree strips bool'
            return self
        elif isinstance(x, AndFilter):
            if 'must_not' in self.subtree:
                x.subtree['must'].append(self.build())
            else:
                x.subtree['must'].append(self.subtree)
            return x
        return AndFilter(self, x)

    def __or__(self, x):
        # Combine results
        if isinstance(self, OrFilter):
            if 'must_not' in x.subtree:
                self.subtree['should'].append(x.build())
            else:
                self.subtree['should'].append(x.subtree)
            return self
        elif isinstance(x, OrFilter):
            if 'must_not' in self.subtree:
                x.subtree['should'].append(self.build())
            else:
                x.subtree['should'].append(self.subtree)
            return x
        return OrFilter(self, x)

    def __invert__(self):
        return NotFilter(self)

    def empty(self):
        if self._filter is None:
            return True
        return False

    def __repr__(self):
        return str(self._filter)

    @property
    def subtree(self):
        if 'bool' in self._filter:
            return self._filter['bool']
        else:
            return self._filter

    def build(self):
        return self._filter


# Binary operator
class AndFilter(BooleanFilter):
    def __init__(self, *args):
        [isinstance(x, BooleanFilter) for x in args]
        super(AndFilter, self).__init__()
        self._filter = {'bool': {'must': [x.build() for x in args]}}


class OrFilter(BooleanFilter):
    def __init__(self, *args):
        [isinstance(x, BooleanFilter) for x in args]
        super(OrFilter, self).__init__()
        self._filter = {'bool': {'should': [x.build() for x in args]}}


class NotFilter(BooleanFilter):
    def __init__(self, x):
        assert isinstance(x, BooleanFilter)
        super(NotFilter, self).__init__()
        self._filter = {'bool': {'must_not': x.build()}}


# LeafBooleanFilter
class GreaterEqual(BooleanFilter):
    def __init__(self, field, value):
        super(GreaterEqual, self).__init__()
        self._filter = {'range': {field: {'gte': value}}}


class Greater(BooleanFilter):
    def __init__(self, field, value):
        super(Greater, self).__init__()
        self._filter = {'range': {field: {'gt': value}}}


class LessEqual(BooleanFilter):
    def __init__(self, field, value):
        super(LessEqual, self).__init__()
        self._filter = {'range': {field: {'lte': value}}}


class Less(BooleanFilter):
    def __init__(self, field, value):
        super(Less, self).__init__()
        self._filter = {'range': {field: {'lt': value}}}


class Equal(BooleanFilter):
    def __init__(self, field, value):
        super(Equal, self).__init__()
        self._filter = {'term': {field: value}}


class IsIn(BooleanFilter):
    def __init__(self, field, value):
        super(IsIn, self).__init__()
        assert isinstance(value, list)
        if field == 'ids':
            self._filter = {'ids': {'values': value}}
        else:
            self._filter = {'terms': {field: value}}


class Like(BooleanFilter):
    def __init__(self, field, value):
        super(Like, self).__init__()
        self._filter = {'wildcard': {field: value}}


class Rlike(BooleanFilter):
    def __init__(self, field, value):
        super(Rlike, self).__init__()
        self._filter = {'regexp': {field: value}}


class Startswith(BooleanFilter):
    def __init__(self, field, value):
        super(Startswith, self).__init__()
        self._filter = {'prefix': {field: value}}


class IsNull(BooleanFilter):
    def __init__(self, field):
        super(IsNull, self).__init__()
        self._filter = {'missing': {'field': field}}


class NotNull(BooleanFilter):
    def __init__(self, field):
        super(NotNull, self).__init__()
        self._filter = {'exists': {'field': field}}


class ScriptFilter(BooleanFilter):
    def __init__(self, inline, lang=None, params=None):
        super(ScriptFilter, self).__init__()
        script = {'inline': inline}
        if lang is not None:
            script['lang'] = lang
        if params is not None:
            script['params'] = params
        self._filter = {'script': {'script': script}}
