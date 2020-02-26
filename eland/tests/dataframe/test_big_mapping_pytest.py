#  Copyright 2020 Elasticsearch BV
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

import eland as ed
from eland.tests.common import ES_TEST_CLIENT
from eland.tests.common import TestData


class TestDataFrameBigMapping(TestData):

    def test_big_mapping(self):
        mapping = {'mappings': {'properties': {}}}

        for i in range(0, 1000):
            field_name = "long_field_name_" + str(i)
            mapping['mappings']['properties'][field_name] = {'type': 'float'}

        ES_TEST_CLIENT.indices.delete(index='thousand_fields', ignore=[400, 404])
        ES_TEST_CLIENT.indices.create(index='thousand_fields', body=mapping)

        ed_df = ed.DataFrame(ES_TEST_CLIENT, 'thousand_fields')
        ed_df.info()

        ES_TEST_CLIENT.indices.delete(index='thousand_fields')
