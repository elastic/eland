# File called _pytest for PyCharm compatability

import pytest

import eland as ed
from eland.tests import ES_TEST_CLIENT
from eland.tests import FLIGHTS_INDEX_NAME


class TestDataFrameInit:

    def test_init(self):
        # Construct empty DataFrame (throws)
        with pytest.raises(ValueError):
            df = ed.DataFrame()

        # Construct invalid DataFrame (throws)
        with pytest.raises(ValueError):
            df = ed.DataFrame(client=ES_TEST_CLIENT)

        # Construct invalid DataFrame (throws)
        with pytest.raises(ValueError):
            df = ed.DataFrame(index_pattern=FLIGHTS_INDEX_NAME)

        # Good constructors
        df0 = ed.DataFrame(ES_TEST_CLIENT, FLIGHTS_INDEX_NAME)
        df1 = ed.DataFrame(client=ES_TEST_CLIENT, index_pattern=FLIGHTS_INDEX_NAME)

        qc = ed.ElandQueryCompiler(client=ES_TEST_CLIENT, index_pattern=FLIGHTS_INDEX_NAME)
        df2 = ed.DataFrame(query_compiler=qc)
