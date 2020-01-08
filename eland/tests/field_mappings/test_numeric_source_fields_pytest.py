#  Copyright 2019 Elasticsearch BV
#
#      Licensed under the Apache License, Version 2.0 (the "License");
#      you may not use this file except in compliance with the License.
#      You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#      Unless required by applicable law or agreed to in writing, software
#      distributed under the License is distributed on an "AS IS" BASIS,
#      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#      See the License for the specific language governing permissions and
#      limitations under the License.

# File called _pytest for PyCharm compatability

import numpy as np

import eland as ed
from eland.tests import ES_TEST_CLIENT, ECOMMERCE_INDEX_NAME, FLIGHTS_INDEX_NAME
from eland.tests.common import TestData


class TestNumericSourceFields(TestData):

    def test_flights_all_numeric_source_fields(self):
        ed_field_mappings = ed.FieldMappings(
            client=ed.Client(ES_TEST_CLIENT),
            index_pattern=FLIGHTS_INDEX_NAME
        )
        pd_flights = self.pd_flights()

        ed_numeric = ed_field_mappings.numeric_source_fields(include_bool=False)
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
        ed_field_mappings = ed.FieldMappings(
            client=ed.Client(ES_TEST_CLIENT),
            index_pattern=ECOMMERCE_INDEX_NAME,
            display_names=field_names
        )
        pd_ecommerce = self.pd_ecommerce()[field_names]

        ed_numeric = ed_field_mappings.numeric_source_fields(include_bool=False)
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
        ed_field_mappings = ed.FieldMappings(
            client=ed.Client(ES_TEST_CLIENT),
            index_pattern=ECOMMERCE_INDEX_NAME,
            display_names=field_names
        )
        pd_ecommerce = self.pd_ecommerce()[field_names]

        ed_numeric = ed_field_mappings.numeric_source_fields(include_bool=False)
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
        ed_field_mappings = ed.FieldMappings(
            client=ed.Client(ES_TEST_CLIENT),
            index_pattern=ECOMMERCE_INDEX_NAME,
            display_names=field_names
        )
        pd_ecommerce = self.pd_ecommerce()[field_names]

        ed_numeric = ed_field_mappings.numeric_source_fields(include_bool=False)
        pd_numeric = pd_ecommerce.select_dtypes(include=np.number)

        assert pd_numeric.columns.to_list() == ed_numeric
