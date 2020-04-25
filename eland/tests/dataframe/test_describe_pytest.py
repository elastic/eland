# Licensed to Elasticsearch B.V under one or more agreements.
# Elasticsearch B.V licenses this file to you under the Apache 2.0 License.
# See the LICENSE file in the project root for more information

# File called _pytest for PyCharm compatability

from pandas.testing import assert_frame_equal

from eland.tests.common import TestData


class TestDataFrameDescribe(TestData):
    def test_flights_describe(self):
        pd_flights = self.pd_flights()
        ed_flights = self.ed_flights()

        pd_describe = pd_flights.describe()
        ed_describe = ed_flights.describe()

        assert_frame_equal(
            pd_describe.drop(["25%", "50%", "75%"], axis="index"),
            ed_describe.drop(["25%", "50%", "75%"], axis="index"),
            check_exact=False,
            check_less_precise=True,
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
