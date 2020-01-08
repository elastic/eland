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
import pytest

import eland as ed
from eland.tests import ES_TEST_CLIENT, ECOMMERCE_INDEX_NAME
from eland.tests.common import TestData


class TestAggregatables(TestData):

    @pytest.mark.filterwarnings("ignore:Aggregations not supported")
    def test_ecommerce_all_aggregatables(self):
        ed_field_mappings = ed.FieldMappings(
            client=ed.Client(ES_TEST_CLIENT),
            index_pattern=ECOMMERCE_INDEX_NAME
        )

        aggregatables = ed_field_mappings.aggregatable_field_names()

        expected = {'category.keyword': 'category',
                    'currency': 'currency',
                    'customer_birth_date': 'customer_birth_date',
                    'customer_first_name.keyword': 'customer_first_name',
                    'customer_full_name.keyword': 'customer_full_name',
                    'customer_id': 'customer_id',
                    'customer_last_name.keyword': 'customer_last_name',
                    'customer_phone': 'customer_phone',
                    'day_of_week': 'day_of_week',
                    'day_of_week_i': 'day_of_week_i',
                    'email': 'email',
                    'geoip.city_name': 'geoip.city_name',
                    'geoip.continent_name': 'geoip.continent_name',
                    'geoip.country_iso_code': 'geoip.country_iso_code',
                    'geoip.location': 'geoip.location',
                    'geoip.region_name': 'geoip.region_name',
                    'manufacturer.keyword': 'manufacturer',
                    'order_date': 'order_date',
                    'order_id': 'order_id',
                    'products._id.keyword': 'products._id',
                    'products.base_price': 'products.base_price',
                    'products.base_unit_price': 'products.base_unit_price',
                    'products.category.keyword': 'products.category',
                    'products.created_on': 'products.created_on',
                    'products.discount_amount': 'products.discount_amount',
                    'products.discount_percentage': 'products.discount_percentage',
                    'products.manufacturer.keyword': 'products.manufacturer',
                    'products.min_price': 'products.min_price',
                    'products.price': 'products.price',
                    'products.product_id': 'products.product_id',
                    'products.product_name.keyword': 'products.product_name',
                    'products.quantity': 'products.quantity',
                    'products.sku': 'products.sku',
                    'products.tax_amount': 'products.tax_amount',
                    'products.taxful_price': 'products.taxful_price',
                    'products.taxless_price': 'products.taxless_price',
                    'products.unit_discount_amount': 'products.unit_discount_amount',
                    'sku': 'sku',
                    'taxful_total_price': 'taxful_total_price',
                    'taxless_total_price': 'taxless_total_price',
                    'total_quantity': 'total_quantity',
                    'total_unique_products': 'total_unique_products',
                    'type': 'type',
                    'user': 'user'}

        assert expected == aggregatables

    def test_ecommerce_selected_aggregatables(self):
        expected = {'category.keyword': 'category',
                    'currency': 'currency',
                    'customer_birth_date': 'customer_birth_date',
                    'customer_first_name.keyword': 'customer_first_name',
                    'type': 'type', 'user': 'user'}

        ed_field_mappings = ed.FieldMappings(
            client=ed.Client(ES_TEST_CLIENT),
            index_pattern=ECOMMERCE_INDEX_NAME,
            display_names=expected.values()
        )

        aggregatables = ed_field_mappings.aggregatable_field_names()

        assert expected == aggregatables

    def test_ecommerce_single_aggregatable_field(self):
        ed_field_mappings = ed.FieldMappings(
            client=ed.Client(ES_TEST_CLIENT),
            index_pattern=ECOMMERCE_INDEX_NAME
        )

        assert 'user' == ed_field_mappings.aggregatable_field_name('user')

    def test_ecommerce_single_keyword_aggregatable_field(self):
        ed_field_mappings = ed.FieldMappings(
            client=ed.Client(ES_TEST_CLIENT),
            index_pattern=ECOMMERCE_INDEX_NAME
        )

        assert 'customer_first_name.keyword' == ed_field_mappings.aggregatable_field_name('customer_first_name')

    def test_ecommerce_single_non_existant_field(self):
        ed_field_mappings = ed.FieldMappings(
            client=ed.Client(ES_TEST_CLIENT),
            index_pattern=ECOMMERCE_INDEX_NAME
        )

        with pytest.raises(KeyError):
            aggregatable = ed_field_mappings.aggregatable_field_name('non_existant')

    @pytest.mark.filterwarnings("ignore:Aggregations not supported")
    def test_ecommerce_single_non_aggregatable_field(self):
        ed_field_mappings = ed.FieldMappings(
            client=ed.Client(ES_TEST_CLIENT),
            index_pattern=ECOMMERCE_INDEX_NAME
        )

        assert None == ed_field_mappings.aggregatable_field_name('customer_gender')
