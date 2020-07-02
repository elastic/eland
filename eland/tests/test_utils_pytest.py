# Licensed to Elasticsearch B.V under one or more agreements.
# Elasticsearch B.V licenses this file to you under the Apache 2.0 License.
# See the LICENSE file in the project root for more information

# File called _pytest for PyCharm compatibility

from eland.utils import is_valid_attr_name


class TestUtils:
    def test_is_valid_attr_name(self):
        assert is_valid_attr_name("_piZZa")
        assert is_valid_attr_name("nice_pizza_with_2_mushrooms")
        assert is_valid_attr_name("_2_pizze")
        assert is_valid_attr_name("_")
        assert is_valid_attr_name("___")

        assert not is_valid_attr_name("4")
        assert not is_valid_attr_name(4)
        assert not is_valid_attr_name(None)
        assert not is_valid_attr_name("4pizze")
        assert not is_valid_attr_name("pizza+")
