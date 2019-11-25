# File called _pytest for PyCharm compatability

import numpy as np

from pandas.util.testing import assert_series_equal

from eland.tests.common import TestData


class TestMappingsNumericSourceFields(TestData):

    def test_flights_numeric_source_fields(self):
        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()

        ed_numeric = ed_flights._query_compiler._mappings.numeric_source_fields(field_names=None, include_bool=False)
        pd_numeric = pd_flights.select_dtypes(include=np.number)

        assert pd_numeric.columns.to_list() == ed_numeric

    def test_ecommerce_selected_non_numeric_source_fields(self):
        field_names = ['category', 'currency', 'customer_birth_date', 'customer_first_name', 'user']
        """
        Note: non of there are numeric
        category                       object
        currency                       object
        customer_birth_date    datetime64[ns]
        customer_first_name            object
        user                           object
        """

        ed_ecommerce = self.ed_ecommerce()[field_names]
        pd_ecommerce = self.pd_ecommerce()[field_names]

        ed_numeric = ed_ecommerce._query_compiler._mappings.numeric_source_fields(field_names=field_names, include_bool=False)
        pd_numeric = pd_ecommerce.select_dtypes(include=np.number)

        assert pd_numeric.columns.to_list() == ed_numeric

    def test_ecommerce_selected_mixed_numeric_source_fields(self):
        field_names = ['category', 'currency', 'customer_birth_date', 'customer_first_name', 'total_quantity', 'user']

        """
        Note: one is numeric
        category                       object
        currency                       object
        customer_birth_date    datetime64[ns]
        customer_first_name            object
        total_quantity                 int64
        user                           object
        """

        ed_ecommerce = self.ed_ecommerce()[field_names]
        pd_ecommerce = self.pd_ecommerce()[field_names]

        ed_numeric = ed_ecommerce._query_compiler._mappings.numeric_source_fields(field_names=field_names, include_bool=False)
        pd_numeric = pd_ecommerce.select_dtypes(include=np.number)

        assert pd_numeric.columns.to_list() == ed_numeric

    def test_ecommerce_selected_all_numeric_source_fields(self):
        field_names = ['total_quantity', 'taxful_total_price', 'taxless_total_price']

        """
        Note: all are numeric
        total_quantity           int64
        taxful_total_price     float64
        taxless_total_price    float64
        """

        ed_ecommerce = self.ed_ecommerce()[field_names]
        pd_ecommerce = self.pd_ecommerce()[field_names]

        ed_numeric = ed_ecommerce._query_compiler._mappings.numeric_source_fields(field_names=field_names, include_bool=False)
        pd_numeric = pd_ecommerce.select_dtypes(include=np.number)

        assert pd_numeric.columns.to_list() == ed_numeric
