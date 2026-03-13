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

"""
Utility functions for PyTorch module that can be shared across modules
without causing circular imports.
"""

from typing import Any, Set, Tuple, Union


def _is_tokenizer_type(
    tokenizer: Any, tokenizer_class_names: Union[str, Tuple[str, ...], Set[str]]
) -> bool:
    """
    Check if tokenizer is one of the specified types by class name.
    Works even if tokenizer classes are not directly importable.

    Args:
        tokenizer: The tokenizer instance to check
        tokenizer_class_names: String or tuple of strings with class names

    Returns:
        bool: True if tokenizer matches any of the specified types
    """
    if isinstance(tokenizer_class_names, str):
        tokenizer_class_names = (tokenizer_class_names,)
    tokenizer_class_name = tokenizer.__class__.__name__
    return tokenizer_class_name in tokenizer_class_names
