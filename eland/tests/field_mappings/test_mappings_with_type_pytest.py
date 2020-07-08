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

# File called _pytest for PyCharm compatability
import pytest

from eland.field_mappings import FieldMappings
from eland.tests.common import TestData


class TestMappingsWithType(TestData):
    def test_mappings_with_type(self):
        # Unless we spin up a 6.x index, this is difficult
        # to test. This is not ideal, but supporting some basic
        # features on 6.x indices makes eland more generally usable.
        #
        # For now, just test function:
        mapping7x = {
            "my_index": {
                "mappings": {
                    "properties": {
                        "city": {
                            "type": "text",
                            "fields": {"keyword": {"type": "keyword"}},
                        }
                    }
                }
            }
        }

        expected7x_source_only_false = {
            "city": ("text", None),
            "city.keyword": ("keyword", None),
        }
        expected7x_source_only_true = {"city": ("text", None)}

        mapping6x = {
            "my_index": {
                "mappings": {
                    "doc": {
                        "properties": {
                            "city": {
                                "type": "text",
                                "fields": {"keyword": {"type": "keyword"}},
                            }
                        }
                    }
                }
            }
        }

        expected6x_source_only_false = {
            "city": ("text", None),
            "city.keyword": ("keyword", None),
        }
        expected6x_source_only_true = {"city": ("text", None)}

        # add a 5x mapping to get coverage of error
        mapping5x = {
            "my_index": {
                "mappings": {
                    "user": {
                        "properties": {
                            "name": {"type": "text"},
                            "user_name": {"type": "keyword"},
                            "email": {"type": "keyword"},
                        }
                    },
                    "tweet": {
                        "properties": {
                            "content": {"type": "text"},
                            "user_name": {"type": "keyword"},
                            "tweeted_at": {"type": "date"},
                        }
                    },
                }
            }
        }

        result7x = FieldMappings._extract_fields_from_mapping(mapping7x)
        assert expected7x_source_only_false == result7x

        result7x = FieldMappings._extract_fields_from_mapping(
            mapping7x, source_only=True
        )
        assert expected7x_source_only_true == result7x

        result6x = FieldMappings._extract_fields_from_mapping(mapping6x)
        assert expected6x_source_only_false == result6x

        result6x = FieldMappings._extract_fields_from_mapping(
            mapping6x, source_only=True
        )
        assert expected6x_source_only_true == result6x

        with pytest.raises(NotImplementedError):
            FieldMappings._extract_fields_from_mapping(mapping5x)
