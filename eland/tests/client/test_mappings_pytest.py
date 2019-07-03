# File called _pytest for PyCharm compatability

import numpy as np
from pandas.util.testing import (
    assert_series_equal, assert_frame_equal)

import eland as ed
from eland.tests import *
from eland.tests.common import TestData


class TestMapping(TestData):

    # Requires 'setup_tests.py' to be run prior to this
    def test_fields(self):
        mappings = ed.Mappings(ed.Client(ELASTICSEARCH_HOST), TEST_MAPPING1_INDEX_NAME)

        assert TEST_MAPPING1_EXPECTED_DF.index.tolist() == mappings.all_fields()

        assert_frame_equal(TEST_MAPPING1_EXPECTED_DF, pd.DataFrame(mappings._mappings_capabilities['es_dtype']))

        assert TEST_MAPPING1_EXPECTED_SOURCE_FIELD_COUNT == mappings.count_source_fields()

    def test_copy(self):
        mappings = ed.Mappings(ed.Client(ELASTICSEARCH_HOST), TEST_MAPPING1_INDEX_NAME)

        assert TEST_MAPPING1_EXPECTED_DF.index.tolist() == mappings.all_fields()
        assert_frame_equal(TEST_MAPPING1_EXPECTED_DF, pd.DataFrame(mappings._mappings_capabilities['es_dtype']))
        assert TEST_MAPPING1_EXPECTED_SOURCE_FIELD_COUNT == mappings.count_source_fields()

        # Pick 1 source field
        columns = ['dest_location']
        mappings_copy1 = ed.Mappings(mappings=mappings, columns=columns)

        assert columns == mappings_copy1.all_fields()
        assert len(columns) == mappings_copy1.count_source_fields()

        # Pick 3 source fields (out of order)
        columns = ['dest_location', 'city', 'user_name']
        mappings_copy2 = ed.Mappings(mappings=mappings, columns=columns)

        assert columns == mappings_copy2.all_fields()
        assert len(columns) == mappings_copy2.count_source_fields()

        # Check original is still ok
        assert TEST_MAPPING1_EXPECTED_DF.index.tolist() == mappings.all_fields()
        assert_frame_equal(TEST_MAPPING1_EXPECTED_DF, pd.DataFrame(mappings._mappings_capabilities['es_dtype']))
        assert TEST_MAPPING1_EXPECTED_SOURCE_FIELD_COUNT == mappings.count_source_fields()

    def test_dtypes(self):
        mappings = ed.Mappings(ed.Client(ELASTICSEARCH_HOST), TEST_MAPPING1_INDEX_NAME)

        expected_dtypes = pd.Series(
            {'city': 'object', 'content': 'object', 'dest_location': 'object', 'email': 'object',
             'maps-telemetry.attributesPerMap.dataSourcesCount.avg': 'int64',
             'maps-telemetry.attributesPerMap.dataSourcesCount.max': 'int64',
             'maps-telemetry.attributesPerMap.dataSourcesCount.min': 'int64',
             'maps-telemetry.attributesPerMap.emsVectorLayersCount.france_departments.avg': 'float64',
             'maps-telemetry.attributesPerMap.emsVectorLayersCount.france_departments.max': 'int64',
             'maps-telemetry.attributesPerMap.emsVectorLayersCount.france_departments.min': 'int64',
             'my_join_field': 'object', 'name': 'object', 'origin_location.lat': 'object',
             'origin_location.lon': 'object', 'text': 'object', 'tweeted_at': 'datetime64[ns]',
             'type': 'object', 'user_name': 'object'})

        assert_series_equal(expected_dtypes, mappings.dtypes())

    def test_get_dtype_counts(self):
        mappings = ed.Mappings(ed.Client(ELASTICSEARCH_HOST), TEST_MAPPING1_INDEX_NAME)

        expected_get_dtype_counts = pd.Series({'datetime64[ns]': 1, 'float64': 1, 'int64': 5, 'object': 11})

        assert_series_equal(expected_get_dtype_counts, mappings.get_dtype_counts())

    def test_mapping_capabilities(self):
        mappings = ed.Mappings(ed.Client(ELASTICSEARCH_HOST), TEST_MAPPING1_INDEX_NAME)

        field_capabilities = mappings.field_capabilities('city')

        assert True == field_capabilities['_source']
        assert 'text' == field_capabilities['es_dtype']
        assert 'object' == field_capabilities['pd_dtype']
        assert True == field_capabilities['searchable']
        assert False == field_capabilities['aggregatable']

        field_capabilities = mappings.field_capabilities('city.raw')

        assert False == field_capabilities['_source']
        assert 'keyword' == field_capabilities['es_dtype']
        assert 'object' == field_capabilities['pd_dtype']
        assert True == field_capabilities['searchable']
        assert True == field_capabilities['aggregatable']

    def test_generate_es_mappings(self):
        df = pd.DataFrame(data={'A': np.random.rand(3),
                                'B': 1,
                                'C': 'foo',
                                'D': pd.Timestamp('20190102'),
                                'E': [1.0, 2.0, 3.0],
                                'F': False,
                                'G': [1, 2, 3]},
                          index=['0','1','2'])

        expected_mappings = {'mappings': {
            'properties': {'A': {'type': 'double'},
                           'B': {'type': 'long'},
                           'C': {'type': 'keyword'},
                           'D': {'type': 'date'},
                           'E': {'type': 'double'},
                           'F': {'type': 'boolean'},
                           'G': {'type': 'long'}}}}

        mappings = ed.Mappings._generate_es_mappings(df)

        assert expected_mappings == mappings

        # Now create index
        index_name = 'eland_test_generate_es_mappings'

        ed.pandas_to_es(df, ELASTICSEARCH_HOST, index_name, if_exists="replace", refresh=True)

        ed_df = ed.DataFrame(ELASTICSEARCH_HOST, index_name)
        ed_df_head = ed_df.head()

        assert_frame_equal(df, ed_df_head)
