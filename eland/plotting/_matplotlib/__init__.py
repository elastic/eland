# Licensed to Elasticsearch B.V under one or more agreements.
# Elasticsearch B.V licenses this file to you under the Apache 2.0 License.
# See the LICENSE file in the project root for more information

"""
Public plotting API

Based from https://github.com/pandas-dev/pandas/blob/v0.25.3/pandas/plotting/__init__.py
but only supporting a subset of plotting methods (for now).
"""

from eland.plotting._matplotlib.hist import (
    hist_frame,
    hist_series,
)

__all__ = [
    "hist_frame",
    "hist_series",
]
