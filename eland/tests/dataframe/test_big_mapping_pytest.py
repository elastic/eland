# Licensed to Elasticsearch B.V under one or more agreements.
# Elasticsearch B.V licenses this file to you under the Apache 2.0 License.
# See the LICENSE file in the project root for more information

# File called _pytest for PyCharm compatability

import eland as ed
from eland.tests.common import ES_TEST_CLIENT
from eland.tests.common import TestData


class TestDataFrameBigMapping(TestData):
    def test_big_mapping(self):
        mapping = {"mappings": {"properties": {}}}

        for i in range(0, 1000):
            field_name = "long_field_name_" + str(i)
            mapping["mappings"]["properties"][field_name] = {"type": "float"}

        ES_TEST_CLIENT.indices.delete(index="thousand_fields", ignore=[400, 404])
        ES_TEST_CLIENT.indices.create(index="thousand_fields", body=mapping)

        ed_df = ed.DataFrame(ES_TEST_CLIENT, "thousand_fields")
        ed_df.info()

        ES_TEST_CLIENT.indices.delete(index="thousand_fields")
