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

from typing import Any, Optional, Sequence

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
