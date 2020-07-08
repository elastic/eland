#  Licensed to Elasticsearch B.V. under one or more contributor
#  license agreements. See the NOTICE file distributed with
#  this work for additional information regarding copyright
#  ownership. Elasticsearch B.V. licenses this file to you under
#  the Apache License, Version 2.0 (the "License"); you may
#  not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
# 	http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the License is distributed on an
#  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
#  specific language governing permissions and limitations
#  under the License.

import re
import functools
import warnings
from typing import Callable, TypeVar


F = TypeVar("F")


def deprecated_api(replace_with: str) -> Callable[[F], F]:
    def wrapper(f: F) -> F:
        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            warnings.warn(
                f"{f.__name__} is deprecated, use {replace_with} instead",
                DeprecationWarning,
            )
            return f(*args, **kwargs)

        return wrapped

    return wrapper


def is_valid_attr_name(s):
    """
    Ensure the given string can be used as attribute on an object instance.
    """
    return isinstance(s, str) and re.search(
        string=s, pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$"
    )
