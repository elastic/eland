# File called _pytest for PyCharm compatability

from pandas.util.testing import assert_series_equal

from eland.tests.common import TestData


class TestMappingsDtypes(TestData):

    def test_flights_dtypes_all(self):
        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()

        pd_dtypes = pd_flights.dtypes
        ed_dtypes = ed_flights._query_compiler._mappings.dtypes()

        assert_series_equal(pd_dtypes, ed_dtypes)

    def test_flights_dtypes_columns(self):
        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()[['Carrier', 'AvgTicketPrice', 'Cancelled']]

        pd_dtypes = pd_flights.dtypes
        ed_dtypes = ed_flights._query_compiler._mappings.dtypes(field_names=['Carrier', 'AvgTicketPrice', 'Cancelled'])

        assert_series_equal(pd_dtypes, ed_dtypes)
