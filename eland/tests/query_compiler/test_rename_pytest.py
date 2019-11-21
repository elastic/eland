# File called _pytest for PyCharm compatability
import pandas as pd

from pandas.util.testing import assert_series_equal

from eland import ElandQueryCompiler
from eland.tests.common import TestData


class TestQueryCompilerRename(TestData):

    def test_query_compiler_basic_rename(self):
        field_names = []
        display_names = []

        mapper = ElandQueryCompiler.DisplayNameToFieldNameMapper()

        assert field_names == mapper.field_names_to_list()
        assert display_names == mapper.display_names_to_list()

        field_names = ['a']
        display_names = ['A']
        update_A = {'a' : 'A'}
        mapper.rename_display_name(update_A)

        assert field_names == mapper.field_names_to_list()
        assert display_names == mapper.display_names_to_list()

        field_names = ['a', 'b']
        display_names = ['A', 'B']

        update_B = {'b' : 'B'}
        mapper.rename_display_name(update_B)

        assert field_names == mapper.field_names_to_list()
        assert display_names == mapper.display_names_to_list()

        field_names = ['a', 'b']
        display_names = ['AA', 'B']

        update_AA = {'A' : 'AA'}
        mapper.rename_display_name(update_AA)

        assert field_names == mapper.field_names_to_list()
        assert display_names == mapper.display_names_to_list()

    def test_query_compiler_basic_rename_columns(self):
        columns = ['a', 'b', 'c', 'd']

        mapper = ElandQueryCompiler.DisplayNameToFieldNameMapper()

        display_names = ['A', 'b', 'c', 'd']
        update_A = {'a' : 'A'}
        mapper.rename_display_name(update_A)

        assert display_names == mapper.field_to_display_names(columns)

        # Invalid update
        display_names = ['A', 'b', 'c', 'd']
        update_ZZ = {'a' : 'ZZ'}
        mapper.rename_display_name(update_ZZ)

        assert display_names == mapper.field_to_display_names(columns)

        display_names = ['AA', 'b', 'c', 'd']
        update_AA = {'A' : 'AA'} # already renamed to 'A'
        mapper.rename_display_name(update_AA)

        assert display_names == mapper.field_to_display_names(columns)

        display_names = ['AA', 'b', 'C', 'd']
        update_AA_C = {'a' : 'AA', 'c' : 'C'} # 'a' rename ignored
        mapper.rename_display_name(update_AA_C)

        assert display_names == mapper.field_to_display_names(columns)
