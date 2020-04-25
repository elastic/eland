# Licensed to Elasticsearch B.V under one or more agreements.
# Elasticsearch B.V licenses this file to you under the Apache 2.0 License.
# See the LICENSE file in the project root for more information

import numpy as np
import pandas as pd
import pytest

import eland as ed

# Fix console size for consistent test results
pd.set_option("display.max_rows", 10)
pd.set_option("display.max_columns", 5)
pd.set_option("display.width", 100)


@pytest.fixture(autouse=True)
def add_imports(doctest_namespace):
    doctest_namespace["np"] = np
    doctest_namespace["pd"] = pd
    doctest_namespace["ed"] = ed
