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

import pandas as pd

from eland.tests.common import TestData, assert_series_equal


class TestSeriesDescribe(TestData):
    def test_series_describe(self):
        ed_df = self.ed_flights_small()
        pd_df = self.pd_flights_small()

        ed_desc = ed_df.AvgTicketPrice.describe()
        pd_desc = pd_df.AvgTicketPrice.describe()

        assert isinstance(ed_desc, pd.Series)
        assert ed_desc.shape == pd_desc.shape
        assert ed_desc.dtype == pd_desc.dtype
        assert ed_desc.index.equals(pd_desc.index)

        # Percentiles calculations vary for Elasticsearch
        assert_series_equal(
            ed_desc[["count", "mean", "std", "min", "max"]],
            pd_desc[["count", "mean", "std", "min", "max"]],
            rtol=0.2,
        )
