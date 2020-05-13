# Licensed to Elasticsearch B.V under one or more agreements.
# Elasticsearch B.V licenses this file to you under the Apache 2.0 License.
# See the LICENSE file in the project root for more information

from typing import Sequence, Optional, Any
from .._model_serializer import ModelSerializer


class ModelTransformer:
    def __init__(
        self,
        model: Any,
        feature_names: Sequence[str],
        classification_labels: Optional[Sequence[str]] = None,
        classification_weights: Optional[Sequence[float]] = None,
    ):
        self._feature_names = feature_names
        self._model = model
        self._classification_labels = classification_labels
        self._classification_weights = classification_weights

    def transform(self) -> ModelSerializer:
        raise NotImplementedError()

    @property
    def model_type(self) -> str:
        raise NotImplementedError()
