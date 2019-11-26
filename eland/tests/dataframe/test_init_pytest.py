# File called _pytest for PyCharm compatability

import pytest

import eland as ed
from eland.tests import ELASTICSEARCH_HOST
from eland.tests import FLIGHTS_INDEX_NAME


class TestDataFrameInit:

    def test_init(self):
        # Construct empty DataFrame (throws)
        with pytest.raises(ValueError):
            df = ed.DataFrame()

        # Construct invalid DataFrame (throws)
        with pytest.raises(ValueError):
            df = ed.DataFrame(client=ELASTICSEARCH_HOST)

        # Construct invalid DataFrame (throws)
        with pytest.raises(ValueError):
            df = ed.DataFrame(index_pattern=FLIGHTS_INDEX_NAME)

        # Good constructors
        df0 = ed.DataFrame(ELASTICSEARCH_HOST, FLIGHTS_INDEX_NAME)
        df1 = ed.DataFrame(client=ELASTICSEARCH_HOST, index_pattern=FLIGHTS_INDEX_NAME)

        qc = ed.ElandQueryCompiler(client=ELASTICSEARCH_HOST, index_pattern=FLIGHTS_INDEX_NAME)
        df2 = ed.DataFrame(query_compiler=qc)
