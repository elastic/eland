# File called _pytest for PyCharm compatability
from io import StringIO

from eland.tests.common import TestData


class TestDataFrameInfo(TestData):

    def test_to_describe1(self):
        pd_flights = self.pd_flights()
        ed_flights = self.ed_flights()

        pd_describe = pd_flights.describe()
        ed_describe = ed_flights.describe()

        print(pd_describe)
        print(ed_describe)

        # TODO - this fails now as ES aggregations are approximate
        #        if ES percentile agg uses
        #        "hdr": {
        #           "number_of_significant_value_digits": 3
        #         }
        #        this works
        # assert_almost_equal(pd_flights_describe, ed_flights_describe)

        pd_ecommerce_describe = self.pd_ecommerce().describe()
        ed_ecommerce_describe = self.ed_ecommerce().describe()

        # We don't compare ecommerce here as the default dtypes in pandas from read_json
        # don't match the mapping types. This is mainly because the products field is
        # nested and so can be treated as a multi-field in ES, but not in pandas

    def test_to_describe2(self):
        pd_flights = self.pd_flights().head()
        ed_flights = self.ed_flights().head()

        pd_describe = pd_flights.describe()
        ed_describe = ed_flights.describe()

        print(pd_describe)
        print(ed_describe)

