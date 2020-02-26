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
from io import StringIO

import eland as ed

from eland.tests import ES_TEST_CLIENT

from eland.tests.common import TestData


class TestDataFrameInfo(TestData):

    def test_flights_info(self):
        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()

        ed_buf = StringIO()
        pd_buf = StringIO()

        # Ignore memory_usage and first line (class name)
        ed_flights.info(buf=ed_buf, memory_usage=False)
        pd_flights.info(buf=pd_buf, memory_usage=False)

        ed_buf_lines = ed_buf.getvalue().split('\n')
        pd_buf_lines = pd_buf.getvalue().split('\n')

        assert pd_buf_lines[1:] == ed_buf_lines[1:]

        # NOTE: info does not work on truncated data frames (e.g. head/tail) TODO

        print(self.ed_ecommerce().info())

    def test_empty_info(self):
        mapping = {'mappings': {'properties': {}}}

        for i in range(0, 10):
            field_name = "field_name_" + str(i)
            mapping['mappings']['properties'][field_name] = {'type': 'float'}

        ES_TEST_CLIENT.indices.delete(index='empty_index', ignore=[400, 404])
        ES_TEST_CLIENT.indices.create(index='empty_index', body=mapping)

        ed_df = ed.DataFrame(ES_TEST_CLIENT, 'empty_index')
        ed_df.info()

        ES_TEST_CLIENT.indices.delete(index='empty_index')
