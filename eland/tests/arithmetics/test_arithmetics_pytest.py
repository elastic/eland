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

from eland.arithmetics import *
from eland.tests.common import TestData


class TestQueryCopy(TestData):

    def test_numeric(self):
        v = 10.0
        s = ArithmeticSeries("name")

        v = 10.0 + 10.0 / v
        v = v / 10.0 - 5.0

        s = (10.0) + (10.0) / s
        s = s / (10.0) - (5.0)

        print(s.resolve())
        print(v)

    def test_string(self):
        v = "first"
        first = ArithmeticSeries("first")
        last = ArithmeticSeries("last")

        v = v + " last"

        s = first + ArithmeticString(" ") + last

        print(s.resolve())
        print(v)

