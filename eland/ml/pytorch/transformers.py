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
Support for and interoperability with HuggingFace transformers and related
libraries such as sentence-transformers.
"""

import json
import os.path
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union

import torch
import transformers
from sentence_transformers import SentenceTransformer
from torch import Tensor, nn
from transformers import (
    AutoConfig,
    AutoModel,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

DEFAULT_OUTPUT_KEY = "sentence_embedding"
SUPPORTED_TASK_TYPES = {
    "fill_mask",
    "ner",
    "text_classification",
    "text_embedding",
    "zero_shot_classification",
}
SUPPORTED_TASK_TYPES_NAMES = ", ".join(sorted(SUPPORTED_TASK_TYPES))
SUPPORTED_TOKENIZERS = (
    transformers.BertTokenizer,
    transformers.DPRContextEncoderTokenizer,
    transformers.DPRQuestionEncoderTokenizer,
    transformers.DistilBertTokenizer,
    transformers.ElectraTokenizer,
    transformers.MobileBertTokenizer,
    transformers.RetriBertTokenizer,
    transformers.SqueezeBertTokenizer,
)
SUPPORTED_TOKENIZERS_NAMES = ", ".join(sorted([str(x) for x in SUPPORTED_TOKENIZERS]))

TracedModelTypes = Union[
    torch.nn.Module,
    torch.ScriptModule,
    torch.jit.ScriptModule,
    torch.jit.TopLevelTracedModule,
]


class _DistilBertWrapper(nn.Module):
    """
    A simple wrapper around DistilBERT model which makes the model inputs
    conform to Elasticsearch's native inference processor interface.
    """

    def __init__(self, model: transformers.PreTrainedModel):
        super().__init__()
        self._model = model
        self.config = model.config

    @staticmethod
    def try_wrapping(model: PreTrainedModel) -> Optional[Any]:
        if isinstance(model.config, transformers.DistilBertConfig):
            return _DistilBertWrapper(model)
        else:
            return model

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor,
        position_ids: Tensor,
    ) -> Tensor:
        """Wrap the input and output to conform to the native process interface."""

        return self._model(input_ids=input_ids, attention_mask=attention_mask)


class _SentenceTransformerWrapper(nn.Module):
    """
    A wrapper around sentence-transformer models to provide pooling,
    normalization and other graph layers that are not defined in the base
    HuggingFace transformer model.
    """

    def __init__(self, model: PreTrainedModel, output_key: str = DEFAULT_OUTPUT_KEY):
        super().__init__()
        self._hf_model = model
        self._st_model = SentenceTransformer(model.config.name_or_path)
        self._output_key = output_key

        self._remove_pooling_layer()
        self._replace_transformer_layer()

    @staticmethod
    def from_pretrained(
        model_id: str, output_key: str = DEFAULT_OUTPUT_KEY
    ) -> Optional[Any]:
        if model_id.startswith("sentence-transformers/"):
            model = AutoModel.from_pretrained(model_id, torchscript=True)
            return _SentenceTransformerWrapper(model, output_key)
        else:
            return None

    def _remove_pooling_layer(self):
        """
        Removes any last pooling layer which is not used to create embeddings.
        Leaving this layer in will cause it to return a NoneType which in turn
        will fail to load in libtorch. Alternatively, we can just use the output
        of the pooling layer as a dummy but this also affects (if only in a
        minor way) the performance of inference, so we're better off removing
        the layer if we can.
        """

        if hasattr(self._hf_model, "pooler"):
            self._hf_model.pooler = None

    def _replace_transformer_layer(self):
        """
        Replaces the HuggingFace Transformer layer in the SentenceTransformer
        modules so we can set it with one that has pooling layer removed and
        was loaded ready for TorchScript export.
        """

        self._st_model._modules["0"].auto_model = self._hf_model

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor,
        position_ids: Tensor,
    ) -> Tensor:
        """Wrap the input and output to conform to the native process interface."""

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
        }

        # remove inputs for specific model types
        if isinstance(self._hf_model.config, transformers.DistilBertConfig):
            del inputs["token_type_ids"]

        return self._st_model(inputs)[self._output_key]


class _DPREncoderWrapper(nn.Module):
    """
    AutoModel loading does not work for DPRContextEncoders, this only exists as
    a workaround. This may never be fixed so this is likely permanent.
    See: https://github.com/huggingface/transformers/issues/13670
    """

    _SUPPORTED_MODELS = {
        transformers.DPRContextEncoder,
        transformers.DPRQuestionEncoder,
    }
    _SUPPORTED_MODELS_NAMES = set([x.__name__ for x in _SUPPORTED_MODELS])

    def __init__(
        self,
        model: Union[transformers.DPRContextEncoder, transformers.DPRQuestionEncoder],
    ):
        super().__init__()
        self._model = model

    @staticmethod
    def from_pretrained(model_id: str) -> Optional[Any]:

        config = AutoConfig.from_pretrained(model_id)

        def is_compatible() -> bool:
            is_dpr_model = config.model_type == "dpr"
            has_architectures = len(config.architectures) == 1
            is_supported_architecture = (
                config.architectures[0] in _DPREncoderWrapper._SUPPORTED_MODELS_NAMES
            )
            return is_dpr_model and has_architectures and is_supported_architecture

        if is_compatible():
            model = getattr(transformers, config.architectures[0]).from_pretrained(
                model_id, torchscript=True
            )
            return _DPREncoderWrapper(model)
        else:
            return None

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor,
        position_ids: Tensor,
    ) -> Tensor:
        """Wrap the input and output to conform to the native process interface."""

        return self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )


class _TraceableModel(ABC):
    """A base class representing a HuggingFace transformer model that can be traced."""

    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        model: Union[
            PreTrainedModel,
            _SentenceTransformerWrapper,
            _DPREncoderWrapper,
            _DistilBertWrapper,
        ],
    ):
        self._tokenizer = tokenizer
        self._model = model

    def classification_labels(self) -> Optional[List[str]]:
        return None

    def quantize(self):
        torch.quantization.quantize_dynamic(
            self._model, {torch.nn.Linear}, dtype=torch.qint8
        )

    def trace(self) -> TracedModelTypes:
        # model needs to be in evaluate mode
        self._model.eval()

        inputs = self._prepare_inputs()
        position_ids = torch.arange(inputs["input_ids"].size(1), dtype=torch.long)

        # Add params when not provided by the tokenizer (e.g. DistilBERT), to conform to BERT interface
        if "token_type_ids" not in inputs:
            inputs["token_type_ids"] = torch.zeros(
                inputs["input_ids"].size(1), dtype=torch.long
            )

        return torch.jit.trace(
            self._model,
            (
                inputs["input_ids"],
                inputs["attention_mask"],
                inputs["token_type_ids"],
                position_ids,
            ),
        )

    @abstractmethod
    def _prepare_inputs(self) -> transformers.BatchEncoding:
        ...


class _TraceableClassificationModel(_TraceableModel, ABC):
    def classification_labels(self) -> Optional[List[str]]:
        id_label_items = self._model.config.id2label.items()
        labels = [v for _, v in sorted(id_label_items, key=lambda kv: kv[0])]

        # Make classes like I-PER into I_PER which fits Java enumerations
        return [label.replace("-", "_") for label in labels]


class _TraceableFillMaskModel(_TraceableModel):
    def _prepare_inputs(self) -> transformers.BatchEncoding:
        return self._tokenizer(
            "Who was Jim Henson?",
            "[MASK] Henson was a puppeteer",
            padding="max_length",
            return_tensors="pt",
        )


class _TraceableNerModel(_TraceableClassificationModel):
    def _prepare_inputs(self) -> transformers.BatchEncoding:
        return self._tokenizer(
            (
                "Hugging Face Inc. is a company based in New York City. "
                "Its headquarters are in DUMBO, therefore very close to the Manhattan Bridge."
            ),
            padding="max_length",
            return_tensors="pt",
        )


class _TraceableTextClassificationModel(_TraceableClassificationModel):
    def _prepare_inputs(self) -> transformers.BatchEncoding:
        return self._tokenizer(
            "This is an example sentence.",
            padding="max_length",
            return_tensors="pt",
        )


class _TraceableTextEmbeddingModel(_TraceableModel):
    def _prepare_inputs(self) -> transformers.BatchEncoding:
        return self._tokenizer(
            "This is an example sentence.",
            padding="max_length",
            return_tensors="pt",
        )


class _TraceableZeroShotClassificationModel(_TraceableClassificationModel):
    def _prepare_inputs(self) -> transformers.BatchEncoding:
        return self._tokenizer(
            "This is an example sentence.",
            "This example is an example.",
            padding="max_length",
            return_tensors="pt",
        )


class TransformerModel:
    def __init__(self, model_id: str, task_type: str, quantize: bool = False):
        self._model_id = model_id
        self._task_type = task_type.replace("-", "_")

        # load Hugging Face model and tokenizer
        # use padding in the tokenizer to ensure max length sequences are used for tracing (at call time)
        #  - see: https://huggingface.co/transformers/serialization.html#dummy-inputs-and-standard-lengths
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            self._model_id,
            use_fast=False,
        )

        # check for a supported tokenizer
        if not isinstance(self._tokenizer, SUPPORTED_TOKENIZERS):
            raise TypeError(
                f"Tokenizer type {self._tokenizer} not supported, must be one of: {SUPPORTED_TOKENIZERS_NAMES}"
            )

        self._traceable_model = self._create_traceable_model()
        if quantize:
            self._traceable_model.quantize()
        self._traced_model = self._traceable_model.trace()
        self._vocab = self._load_vocab()
        self._config = self._create_config()

    def _load_vocab(self):
        vocab_items = self._tokenizer.get_vocab().items()
        vocabulary = [k for k, _ in sorted(vocab_items, key=lambda kv: kv[1])]
        return {
            "vocabulary": vocabulary,
        }

    def _create_config(self):
        inference_config = {
            self._task_type: {
                "tokenization": {
                    "bert": {
                        "do_lower_case": getattr(
                            self._tokenizer, "do_lower_case", False
                        ),
                    }
                }
            }
        }

        if hasattr(self._tokenizer, "max_model_input_sizes"):
            max_sequence_length = self._tokenizer.max_model_input_sizes.get(
                self._model_id
            )
            if max_sequence_length:
                inference_config[self._task_type]["tokenization"]["bert"][
                    "max_sequence_length"
                ] = max_sequence_length

        if self._traceable_model.classification_labels():
            inference_config[self._task_type][
                "classification_labels"
            ] = self._traceable_model.classification_labels()

        return {
            "description": f"Model {self._model_id} for task type '{self._task_type}'",
            "model_type": "pytorch",
            "inference_config": inference_config,
            "input": {
                "field_names": ["text_field"],
            },
        }

    def _create_traceable_model(self) -> _TraceableModel:
        if self._task_type == "fill_mask":
            model = transformers.AutoModelForMaskedLM.from_pretrained(
                self._model_id, torchscript=True
            )
            model = _DistilBertWrapper.try_wrapping(model)
            return _TraceableFillMaskModel(self._tokenizer, model)

        elif self._task_type == "ner":
            model = transformers.AutoModelForTokenClassification.from_pretrained(
                self._model_id, torchscript=True
            )
            model = _DistilBertWrapper.try_wrapping(model)
            return _TraceableNerModel(self._tokenizer, model)

        elif self._task_type == "text_classification":
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                self._model_id, torchscript=True
            )
            model = _DistilBertWrapper.try_wrapping(model)
            return _TraceableTextClassificationModel(self._tokenizer, model)

        elif self._task_type == "text_embedding":
            model = _SentenceTransformerWrapper.from_pretrained(self._model_id)
            if not model:
                model = _DPREncoderWrapper.from_pretrained(self._model_id)
            if not model:
                model = transformers.AutoModel.from_pretrained(
                    self._model_id, torchscript=True
                )
            return _TraceableTextEmbeddingModel(self._tokenizer, model)

        elif self._task_type == "zero_shot_classification":
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                self._model_id, torchscript=True
            )
            model = _DistilBertWrapper.try_wrapping(model)
            return _TraceableZeroShotClassificationModel(self._tokenizer, model)

        else:
            raise TypeError(
                f"Unknown task type {self._task_type}, must be one of: {SUPPORTED_TASK_TYPES_NAMES}"
            )

    def elasticsearch_model_id(self):
        # Elasticsearch model IDs need to be a specific format: no special chars, all lowercase, max 64 chars
        return self._model_id.replace("/", "__").lower()[:64]

    def save(self, path: str) -> Tuple[str, str, str]:
        # save traced model
        model_path = os.path.join(path, "traced_pytorch_model.pt")
        torch.jit.save(self._traced_model, model_path)

        # save configuration
        config_path = os.path.join(path, "config.json")
        with open(config_path, "w") as outfile:
            json.dump(self._config, outfile)

        # save vocabulary
        vocab_path = os.path.join(path, "vocabulary.json")
        with open(vocab_path, "w") as outfile:
            json.dump(self._vocab, outfile)

        return model_path, config_path, vocab_path
