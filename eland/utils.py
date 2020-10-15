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

import functools
import re
import warnings
from collections.abc import Collection as ABCCollection
from typing import Any, Callable, Collection, Iterable, List, TypeVar, Union, cast

import pandas as pd  # type: ignore

RT = TypeVar("RT")


def deprecated_api(
    replace_with: str,
) -> Callable[[Callable[..., RT]], Callable[..., RT]]:
    def wrapper(f: Callable[..., RT]) -> Callable[..., RT]:
        @functools.wraps(f)
        def wrapped(*args: Any, **kwargs: Any) -> RT:
            warnings.warn(
                f"{f.__name__} is deprecated, use {replace_with} instead",
                DeprecationWarning,
                stacklevel=2,
            )
            return f(*args, **kwargs)

        return wrapped

    return wrapper


def is_valid_attr_name(s: str) -> bool:
    """
    Ensure the given string can be used as attribute on an object instance.
    """
    return bool(
        isinstance(s, str) and re.search(string=s, pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$")
    )


def to_list(x: Union[Collection[Any], pd.Series]) -> List[Any]:
    if isinstance(x, ABCCollection):
        return list(x)
    elif isinstance(x, pd.Series):
        return cast(List[Any], x.to_list())
    raise NotImplementedError(f"Could not convert {type(x).__name__} into a list")


def try_sort(iterable: Iterable[str]) -> Iterable[str]:
    # Pulled from pandas.core.common since
    # it was deprecated and removed in 1.1
    listed = list(iterable)
    try:
        return sorted(listed)
    except TypeError:
        return listed
