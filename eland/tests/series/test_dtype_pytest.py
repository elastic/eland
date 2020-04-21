import numpy as np
import warnings
from eland.common import build_pd_series, EMPTY_SERIES_DTYPE


def test_empty_series_dtypes():
    with warnings.catch_warnings(record=True) as w:
        s = build_pd_series({})
    assert s.dtype == EMPTY_SERIES_DTYPE
    assert w == []

    # Ensure that a passed-in dtype isn't ignore
    # even if the result is empty.
    with warnings.catch_warnings(record=True) as w:
        s = build_pd_series({}, dtype=np.int32)
    assert np.int32 != EMPTY_SERIES_DTYPE
    assert s.dtype == np.int32
    assert w == []
