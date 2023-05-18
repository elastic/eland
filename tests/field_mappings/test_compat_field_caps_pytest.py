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
from unittest import mock

from eland.field_mappings import _compat_field_caps
from tests.common import TestData


class TestCompatFieldCaps(TestData):
    def test_query_for_es_8_4_4(self):
        # Elasticsearch server <8.5.0 should use the raw perform_request()
        client = mock.Mock()
        client._eland_es_version = (8, 4, 4)

        _compat_field_caps(client, fields="*", index="test-index")

        client.perform_request.assert_called_with(
            "POST",
            "/test-index/_field_caps",
            params={"fields": "*"},
            headers={"accept": "application/json"},
        )

        _compat_field_caps(client, fields="*")

        client.perform_request.assert_called_with(
            "POST",
            "/_field_caps",
            params={"fields": "*"},
            headers={"accept": "application/json"},
        )

    def test_query_for_es_8_5_0(self):
        # Elasticsearch server >=8.5.0 should use the client API.
        client = mock.Mock()
        client._eland_es_version = (8, 5, 0)

        _compat_field_caps(client, fields="*", index="test-index")

        client.field_caps.assert_called_with(fields="*", index="test-index")

        _compat_field_caps(client, fields="*")

        client.field_caps.assert_called_with(fields="*", index=None)
