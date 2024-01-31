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

import os.path
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import torch  # type: ignore
from torch import nn

TracedModelTypes = Union[
    torch.nn.Module,
    torch.ScriptModule,
    torch.jit.ScriptModule,
    torch.jit.TopLevelTracedModule,
]


class TraceableModel(ABC):
    """A base class representing a pytorch model that can be traced."""

    def __init__(
        self,
        model: nn.Module,
    ):
        self._model = model

    def quantize(self) -> None:
        torch.quantization.quantize_dynamic(
            self._model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True
        )

    def trace(self) -> TracedModelTypes:
        # model needs to be in evaluate mode
        self._model.eval()
        return self._trace()

    @abstractmethod
    def sample_output(self) -> torch.Tensor: ...

    @abstractmethod
    def _trace(self) -> TracedModelTypes: ...

    def classification_labels(self) -> Optional[List[str]]:
        return None

    def save(self, path: str) -> str:
        model_path = os.path.join(path, "traced_pytorch_model.pt")
        trace_model = self.trace()
        trace_model = torch.jit.freeze(trace_model)
        torch.jit.save(trace_model, model_path)
        return model_path

    @property
    def model(self) -> nn.Module:
        return self._model
