# Licensed to Elasticsearch B.V under one or more agreements.
# Elasticsearch B.V licenses this file to you under the Apache 2.0 License.
# See the LICENSE file in the project root for more information

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
