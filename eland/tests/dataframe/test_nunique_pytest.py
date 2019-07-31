# File called _pytest for PyCharm compatability
import pandas as pd
import eland as ed

from eland.tests.common import TestData
from eland.tests.common import (
    assert_pandas_eland_frame_equal,
    assert_pandas_eland_series_equal
)

import numpy as np

class TestDataFrameNUnique(TestData):

    def test_nunique1(self):
        ed_ecommerce = self.ed_ecommerce()
        pd_ecommerce = self.pd_ecommerce()

        print(pd_ecommerce.dtypes)
        print(ed_ecommerce.dtypes)
        #ed_nunique = ed_ecommerce.nunique()
        pd_selection = pd_ecommerce.drop(columns=['category'])
        pd_nunique = pd_selection.nunique(axis=1)

        print(pd_nunique, type(pd_nunique))
