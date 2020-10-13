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

import numpy as np
import pandas as pd
import warnings
from eland.common import build_pd_series, EMPTY_SERIES_DTYPE
from eland.tests.common import assert_series_equal


def test_empty_series_dtypes():
    with warnings.catch_warnings(record=True) as w:
        s = build_pd_series({})
    assert s.dtype == EMPTY_SERIES_DTYPE
    assert w == []

    # Ensure that a passed-in dtype isn't ignore
    # even if the result is empty.
    with warnings.catch_warnings(record=True) as w:
        s = build_pd_series({}, dtype=np.int32)
    assert np.int32 != EMPTY_SERIES_DTYPE
    assert s.dtype == np.int32
    assert w == []


def test_series_es_dtypes(testdata):
    series = testdata.ed_flights_small().AvgTicketPrice
    assert_series_equal(series.es_dtypes, pd.Series(data={"AvgTicketPrice": "float"}))
    assert series.es_dtype == "float"
