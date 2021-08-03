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

from pandas.testing import assert_frame_equal

from tests.common import TestData


class TestDataFrameDescribe(TestData):
    def test_flights_describe(self):
        pd_flights = self.pd_flights()
        ed_flights = self.ed_flights()

        pd_describe = pd_flights.describe()
        # We remove bool columns to match pandas output
        ed_describe = ed_flights.describe().drop(
            ["Cancelled", "FlightDelay"], axis="columns"
        )

        assert_frame_equal(
            pd_describe.drop(["25%", "50%", "75%"], axis="index"),
            ed_describe.drop(["25%", "50%", "75%"], axis="index"),
            check_exact=False,
            rtol=True,
        )

        # TODO - this fails for percentile fields as ES aggregations are approximate
        #        if ES percentile agg uses
        #        "hdr": {
        #           "number_of_significant_value_digits": 3
        #         }
        #        this works

        # pd_ecommerce_describe = self.pd_ecommerce().describe()
        # ed_ecommerce_describe = self.ed_ecommerce().describe()
        # We don't compare ecommerce here as the default dtypes in pandas from read_json
        # don't match the mapping types. This is mainly because the products field is
        # nested and so can be treated as a multi-field in ES, but not in pandas

        # We can not also run 'describe' on a truncate ed dataframe
