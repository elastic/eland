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
