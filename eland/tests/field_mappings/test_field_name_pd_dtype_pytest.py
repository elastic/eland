# Licensed to Elasticsearch B.V under one or more agreements.
# Elasticsearch B.V licenses this file to you under the Apache 2.0 License.
# See the LICENSE file in the project root for more information

# File called _pytest for PyCharm compatability
import pytest
from pandas.testing import assert_series_equal

from eland.field_mappings import FieldMappings
from eland.tests import FLIGHTS_INDEX_NAME, FLIGHTS_MAPPING
from eland.tests.common import ES_TEST_CLIENT
from eland.tests.common import TestData


class TestFieldNamePDDType(TestData):
    def test_all_formats(self):
        ed_field_mappings = FieldMappings(
            client=ES_TEST_CLIENT, index_pattern=FLIGHTS_INDEX_NAME
        )

        pd_flights = self.pd_flights()

        assert_series_equal(pd_flights.dtypes, ed_field_mappings.dtypes())

        for es_field_name in FLIGHTS_MAPPING["mappings"]["properties"].keys():
            pd_dtype = ed_field_mappings.field_name_pd_dtype(es_field_name)

            assert pd_flights[es_field_name].dtype == pd_dtype

    def test_non_existant(self):
        ed_field_mappings = FieldMappings(
            client=ES_TEST_CLIENT, index_pattern=FLIGHTS_INDEX_NAME
        )

        with pytest.raises(KeyError):
            ed_field_mappings.field_name_pd_dtype("unknown")
