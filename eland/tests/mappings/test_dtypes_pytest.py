# File called _pytest for PyCharm compatability

from eland.tests.common import TestData

from pandas.util.testing import assert_series_equal

class TestMappingsDtypes(TestData):

    def test_dtypes1(self):
        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()

        pd_dtypes = pd_flights.dtypes
        ed_dtypes = ed_flights._query_compiler._mappings.dtypes()

        assert_series_equal(pd_dtypes, ed_dtypes)

    def test_dtypes2(self):
        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()[['Carrier', 'AvgTicketPrice', 'Cancelled']]

        pd_dtypes = pd_flights.dtypes
        ed_dtypes = ed_flights._query_compiler._mappings.dtypes(columns=['Carrier', 'AvgTicketPrice', 'Cancelled'])

        assert_series_equal(pd_dtypes, ed_dtypes)

    def test_get_dtype_counts1(self):
        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()

        pd_dtypes = pd_flights.get_dtype_counts().sort_index()
        ed_dtypes = ed_flights._query_compiler._mappings.get_dtype_counts().sort_index()

        assert_series_equal(pd_dtypes, ed_dtypes)

    def test_get_dtype_counts2(self):
        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()[['Carrier', 'AvgTicketPrice', 'Cancelled']]

        pd_dtypes = pd_flights.get_dtype_counts().sort_index()
        ed_dtypes = ed_flights._query_compiler._mappings.\
            get_dtype_counts(columns=['Carrier', 'AvgTicketPrice', 'Cancelled']).sort_index()

        assert_series_equal(pd_dtypes, ed_dtypes)
