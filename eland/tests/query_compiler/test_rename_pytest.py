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

# File called _pytest for PyCharm compatability

from eland import QueryCompiler
from eland.tests.common import TestData


class TestQueryCompilerRename(TestData):

    def test_query_compiler_basic_rename(self):
        field_names = []
        display_names = []

        mapper = QueryCompiler.DisplayNameToFieldNameMapper()

        assert field_names == mapper.field_names_to_list()
        assert display_names == mapper.display_names_to_list()

        field_names = ['a']
        display_names = ['A']
        update_A = {'a': 'A'}
        mapper.rename_display_name(update_A)

        assert field_names == mapper.field_names_to_list()
        assert display_names == mapper.display_names_to_list()

        field_names = ['a', 'b']
        display_names = ['A', 'B']

        update_B = {'b': 'B'}
        mapper.rename_display_name(update_B)

        assert field_names == mapper.field_names_to_list()
        assert display_names == mapper.display_names_to_list()

        field_names = ['a', 'b']
        display_names = ['AA', 'B']

        update_AA = {'A': 'AA'}
        mapper.rename_display_name(update_AA)

        assert field_names == mapper.field_names_to_list()
        assert display_names == mapper.display_names_to_list()

    def test_query_compiler_basic_rename_columns(self):
        columns = ['a', 'b', 'c', 'd']

        mapper = QueryCompiler.DisplayNameToFieldNameMapper()

        display_names = ['A', 'b', 'c', 'd']
        update_A = {'a': 'A'}
        mapper.rename_display_name(update_A)

        assert display_names == mapper.field_to_display_names(columns)

        # Invalid update
        display_names = ['A', 'b', 'c', 'd']
        update_ZZ = {'a': 'ZZ'}
        mapper.rename_display_name(update_ZZ)

        assert display_names == mapper.field_to_display_names(columns)

        display_names = ['AA', 'b', 'c', 'd']
        update_AA = {'A': 'AA'}  # already renamed to 'A'
        mapper.rename_display_name(update_AA)

        assert display_names == mapper.field_to_display_names(columns)

        display_names = ['AA', 'b', 'C', 'd']
        update_AA_C = {'a': 'AA', 'c': 'C'}  # 'a' rename ignored
        mapper.rename_display_name(update_AA_C)

        assert display_names == mapper.field_to_display_names(columns)
