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

import numpy as np
import pytest

from eland.tests.common import TestData


class TestDisplayNames(TestData):

    def test_set_display_names(self):
        pd_flights = self.pd_flights()
        ed_flights = self.ed_flights()

        pd_display_names = pd_flights[['AvgTicketPrice', 'Cancelled']]
        ed_display_names = ed_flights[['AvgTicketPrice', 'Cancelled']]

        buf = StringIO()
        ed_display_names._query_compiler._mappings.info_es(buf)
        print(buf.getvalue())

    def test_set_invalid_display_names(self):
        pd_flights = self.pd_flights()
        ed_flights = self.ed_flights()

        with pytest.raises(KeyError):
            pd_display_names = pd_flights[['non_existant', 'Cancelled']]
        with pytest.raises(KeyError):
            ed_display_names = ed_flights[['non_existant', 'Cancelled']]

