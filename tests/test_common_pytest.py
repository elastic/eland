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

import unittest.mock as mock
import warnings

import pytest

import eland
from eland.common import es_version


@pytest.mark.parametrize(
    ["version_number", "version_tuple"],
    [("7.10.0-alpha1", (7, 10, 0)), ("99.99.99", (99, 99, 99))],
)
def test_major_version_mismatch(version_number, version_tuple):
    client = mock.Mock(spec=["info"])
    client.info.return_value = {"version": {"number": version_number}}
    with warnings.catch_warnings(record=True) as w:
        assert es_version(client) == version_tuple
    assert len(w) == 1
    assert str(w[0].message) == (
        f"Eland major version ({eland.__version__}) doesn't match the major version of the Elasticsearch server ({version_number}) "
        "which can lead to compatibility issues. Your Eland major version should be the same as your cluster major version."
    )
