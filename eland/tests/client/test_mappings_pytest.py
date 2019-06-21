# File called _pytest for PyCharm compatability
import pytest

from eland.tests import *

from pandas.util.testing import (
    assert_almost_equal, assert_frame_equal, assert_series_equal)

import eland as ed

class TestMapping():

    # Requires 'setup_tests.py' to be run prior to this
    def test_mapping(self):
        mapping = ed.Mappings(ed.Client(ELASTICSEARCH_HOST), TEST_MAPPING1_INDEX_NAME)

        assert mapping.all_fields() == TEST_MAPPING1_EXPECTED_DF.index.tolist()

        assert_frame_equal(TEST_MAPPING1_EXPECTED_DF, pd.DataFrame(mapping.mappings_capabilities['es_dtype']))



