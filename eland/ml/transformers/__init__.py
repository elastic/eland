# Licensed to Elasticsearch B.V under one or more agreements.
# Elasticsearch B.V licenses this file to you under the Apache 2.0 License.
# See the LICENSE file in the project root for more information

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
        f"ML model of type {type(model)}, not currently implemented"
    )


try:
    from .sklearn import (
        SKLearnDecisionTreeTransformer,
        SKLearnForestClassifierTransformer,
        SKLearnForestRegressorTransformer,
        SKLearnForestTransformer,
        SKLearnTransformer,
        _MODEL_TRANSFORMERS as _SKLEARN_MODEL_TRANSFORMERS,
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
    from .xgboost import (
        XGBoostClassifierTransformer,
        XGBClassifier,
        XGBoostForestTransformer,
        XGBoostRegressorTransformer,
        XGBRegressor,
        _MODEL_TRANSFORMERS as _XGBOOST_MODEL_TRANSFORMERS,
    )

    __all__ += [
        "XGBoostClassifierTransformer",
        "XGBClassifier",
        "XGBoostForestTransformer",
        "XGBoostRegressorTransformer",
        "XGBRegressor",
    ]
    _MODEL_TRANSFORMERS.update(_XGBOOST_MODEL_TRANSFORMERS)
except ImportError:
    pass
