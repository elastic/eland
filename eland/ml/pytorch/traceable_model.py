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
from typing import List, Optional, Tuple, Union

import torch  # type: ignore
import transformers
from torch import Tensor, nn
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

from eland.ml.pytorch.wrappers import (
    _DistilBertWrapper,
    _DPREncoderWrapper,
    _QuestionAnsweringWrapperModule,
    _SentenceTransformerWrapperModule,
)

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


class TransformerTraceableModel(TraceableModel):
    """A base class representing a HuggingFace transformer model that can be traced."""

    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        model: Union[
            PreTrainedModel,
            _SentenceTransformerWrapperModule,
            _DPREncoderWrapper,
            _DistilBertWrapper,
            _QuestionAnsweringWrapperModule,
        ],
    ):
        super(TransformerTraceableModel, self).__init__(model=model)
        self._tokenizer = tokenizer

    def _trace(self) -> TracedModelTypes:
        inputs = self._compatible_inputs()
        return torch.jit.trace(self._model, example_inputs=inputs)

    def sample_output(self) -> Tensor:
        inputs = self._compatible_inputs()
        return self._model(*inputs)

    def _compatible_inputs(self) -> Tuple[Tensor, ...]:
        inputs = self._prepare_inputs()

        # Add params when not provided by the tokenizer (e.g. DistilBERT), to conform to BERT interface
        if "token_type_ids" not in inputs:
            inputs["token_type_ids"] = torch.zeros(
                inputs["input_ids"].size(1), dtype=torch.long
            )
        if isinstance(
            self._tokenizer,
            (
                transformers.BartTokenizer,
                transformers.MPNetTokenizer,
                transformers.RobertaTokenizer,
                transformers.XLMRobertaTokenizer,
            ),
        ):
            return (inputs["input_ids"], inputs["attention_mask"])

        if isinstance(self._tokenizer, transformers.DebertaV2Tokenizer):
            return (
                inputs["input_ids"],
                inputs["attention_mask"],
                inputs["token_type_ids"],
            )

        position_ids = torch.arange(inputs["input_ids"].size(1), dtype=torch.long)
        inputs["position_ids"] = position_ids
        return (
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["token_type_ids"],
            inputs["position_ids"],
        )

    @abstractmethod
    def _prepare_inputs(self) -> transformers.BatchEncoding: ...


class TraceableClassificationModel(TransformerTraceableModel, ABC):
    def classification_labels(self) -> Optional[List[str]]:
        id_label_items = self._model.config.id2label.items()
        labels = [v for _, v in sorted(id_label_items, key=lambda kv: kv[0])]

        # Make classes like I-PER into I_PER which fits Java enumerations
        return [label.replace("-", "_") for label in labels]


class TraceableFillMaskModel(TransformerTraceableModel):
    def _prepare_inputs(self) -> transformers.BatchEncoding:
        return self._tokenizer(
            "Who was Jim Henson?",
            "[MASK] Henson was a puppeteer",
            padding="max_length",
            return_tensors="pt",
        )


class TraceableTextExpansionModel(TransformerTraceableModel):
    def _prepare_inputs(self) -> transformers.BatchEncoding:
        return self._tokenizer(
            "This is an example sentence.",
            padding="max_length",
            return_tensors="pt",
        )


class TraceableNerModel(TraceableClassificationModel):
    def _prepare_inputs(self) -> transformers.BatchEncoding:
        return self._tokenizer(
            (
                "Hugging Face Inc. is a company based in New York City. "
                "Its headquarters are in DUMBO, therefore very close to the Manhattan Bridge."
            ),
            padding="max_length",
            return_tensors="pt",
        )


class TraceablePassThroughModel(TransformerTraceableModel):
    def _prepare_inputs(self) -> transformers.BatchEncoding:
        return self._tokenizer(
            "This is an example sentence.",
            padding="max_length",
            return_tensors="pt",
        )


class TraceableTextClassificationModel(TraceableClassificationModel):
    def _prepare_inputs(self) -> transformers.BatchEncoding:
        return self._tokenizer(
            "This is an example sentence.",
            padding="max_length",
            return_tensors="pt",
        )


class TraceableTextEmbeddingModel(TransformerTraceableModel):
    def _prepare_inputs(self) -> transformers.BatchEncoding:
        return self._tokenizer(
            "This is an example sentence.",
            padding="longest",
            return_tensors="pt",
        )


class TraceableZeroShotClassificationModel(TraceableClassificationModel):
    def _prepare_inputs(self) -> transformers.BatchEncoding:
        return self._tokenizer(
            "This is an example sentence.",
            "This example is an example.",
            padding="max_length",
            return_tensors="pt",
        )


class TraceableQuestionAnsweringModel(TransformerTraceableModel):
    def _prepare_inputs(self) -> transformers.BatchEncoding:
        return self._tokenizer(
            "What is the meaning of life?"
            "The meaning of life, according to the hitchikers guide, is 42.",
            padding="max_length",
            return_tensors="pt",
        )


class TraceableTextSimilarityModel(TransformerTraceableModel):
    def _prepare_inputs(self) -> transformers.BatchEncoding:
        return self._tokenizer(
            "What is the meaning of life?"
            "The meaning of life, according to the hitchikers guide, is 42.",
            padding="max_length",
            return_tensors="pt",
        )
