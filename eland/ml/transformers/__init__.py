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

import inspect
from typing import Any, Dict, Type

from .base import ModelTransformer

__all__ = ["get_model_transformer"]
_MODEL_TRANSFORMERS: Dict[type, Type[ModelTransformer]] = {}


def get_model_transformer(model: Any, **kwargs: Any) -> ModelTransformer:
    """Creates a ModelTransformer for a given model or raises an exception if one is not available"""
    for model_type, transformer in _MODEL_TRANSFORMERS.items():
        if isinstance(model, model_type):
            # Filter out kwargs that aren't applicable to the specific 'ModelTransformer'
            accepted_kwargs = {
                param for param in inspect.signature(transformer.__init__).parameters
            }
            kwargs = {k: v for k, v in kwargs.items() if k in accepted_kwargs}

            return transformer(model, **kwargs)

    raise NotImplementedError(
        f"Importing ML models of type {type(model)}, not currently implemented"
    )


try:
    from .sklearn import _MODEL_TRANSFORMERS as _SKLEARN_MODEL_TRANSFORMERS
    from .sklearn import (
        SKLearnDecisionTreeTransformer,
        SKLearnForestClassifierTransformer,
        SKLearnForestRegressorTransformer,
        SKLearnForestTransformer,
        SKLearnTransformer,
    )

    __all__ += [
        "SKLearnDecisionTreeTransformer",
        "SKLearnForestClassifierTransformer",
        "SKLearnForestRegressorTransformer",
        "SKLearnForestTransformer",
        "SKLearnTransformer",
    ]
    _MODEL_TRANSFORMERS.update(_SKLEARN_MODEL_TRANSFORMERS)
except ImportError:
    pass

try:
    from .xgboost import _MODEL_TRANSFORMERS as _XGBOOST_MODEL_TRANSFORMERS
    from .xgboost import (
        XGBoostClassifierTransformer,
        XGBoostForestTransformer,
        XGBoostRegressorTransformer,
    )

    __all__ += [
        "XGBoostClassifierTransformer",
        "XGBoostForestTransformer",
        "XGBoostRegressorTransformer",
    ]
    _MODEL_TRANSFORMERS.update(_XGBOOST_MODEL_TRANSFORMERS)
except ImportError:
    pass

try:
    from .lightgbm import _MODEL_TRANSFORMERS as _LIGHTGBM_MODEL_TRANSFORMERS
    from .lightgbm import (
        LGBMClassifierTransformer,
        LGBMForestTransformer,
        LGBMRegressorTransformer,
    )

    __all__ += [
        "LGBMForestTransformer",
        "LGBMRegressorTransformer",
        "LGBMClassifierTransformer",
    ]
    _MODEL_TRANSFORMERS.update(_LIGHTGBM_MODEL_TRANSFORMERS)
except ImportError:
    pass
