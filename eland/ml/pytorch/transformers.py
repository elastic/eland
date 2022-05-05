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
from typing import Any, Dict, List, Optional, Tuple, Union

import torch  # type: ignore
import transformers  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
from torch import Tensor, nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForQuestionAnswering,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from eland.ml.pytorch.nlp_ml_model import (
    FillMaskInferenceOptions,
    NerInferenceOptions,
    NlpBertTokenizationConfig,
    NlpMPNetTokenizationConfig,
    NlpRobertaTokenizationConfig,
    NlpTokenizationConfig,
    NlpTrainedModelConfig,
    PassThroughInferenceOptions,
    QuestionAnsweringInferenceOptions,
    TextClassificationInferenceOptions,
    TextEmbeddingInferenceOptions,
    TrainedModelInput,
    ZeroShotClassificationInferenceOptions,
)
from eland.ml.pytorch.traceable_model import TraceableModel

DEFAULT_OUTPUT_KEY = "sentence_embedding"
SUPPORTED_TASK_TYPES = {
    "fill_mask",
    "ner",
    "text_classification",
    "text_embedding",
    "zero_shot_classification",
    "question_answering",
}
TASK_TYPE_TO_INFERENCE_CONFIG = {
    "fill_mask": FillMaskInferenceOptions,
    "ner": NerInferenceOptions,
    "text_classification": TextClassificationInferenceOptions,
    "text_embedding": TextEmbeddingInferenceOptions,
    "zero_shot_classification": ZeroShotClassificationInferenceOptions,
    "pass_through": PassThroughInferenceOptions,
    "question_answering": QuestionAnsweringInferenceOptions,
}
SUPPORTED_TASK_TYPES_NAMES = ", ".join(sorted(SUPPORTED_TASK_TYPES))
SUPPORTED_TOKENIZERS = (
    transformers.BertTokenizer,
    transformers.MPNetTokenizer,
    transformers.DPRContextEncoderTokenizer,
    transformers.DPRQuestionEncoderTokenizer,
    transformers.DistilBertTokenizer,
    transformers.ElectraTokenizer,
    transformers.MobileBertTokenizer,
    transformers.RetriBertTokenizer,
    transformers.RobertaTokenizer,
    transformers.BartTokenizer,
    transformers.SqueezeBertTokenizer,
)
SUPPORTED_TOKENIZERS_NAMES = ", ".join(sorted([str(x) for x in SUPPORTED_TOKENIZERS]))

TracedModelTypes = Union[
    torch.nn.Module,
    torch.ScriptModule,
    torch.jit.ScriptModule,
    torch.jit.TopLevelTracedModule,
]


class _QuestionAnsweringWrapperModule(nn.Module):  # type: ignore
    """
    A wrapper around a question answering model.
    Our inference engine only takes the first tuple if the inference response
    is a tuple.

    This wrapper transforms the output to be a stacked tensor if its a tuple.

    Otherwise it passes it through
    """

    def __init__(self, model: PreTrainedModel):
        super().__init__()
        self._hf_model = model
        self.config = model.config

    @staticmethod
    def from_pretrained(model_id: str) -> Optional[Any]:
        model = AutoModelForQuestionAnswering.from_pretrained(
            model_id, torchscript=True
        )
        if isinstance(
            model.config,
            (
                transformers.MPNetConfig,
                transformers.RobertaConfig,
                transformers.BartConfig,
            ),
        ):
            return _TwoParameterQuestionAnsweringWrapper(model)
        else:
            return _QuestionAnsweringWrapper(model)


class _QuestionAnsweringWrapper(_QuestionAnsweringWrapperModule):
    def __init__(self, model: PreTrainedModel):
        super().__init__(model=model)

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
            del inputs["position_ids"]
        response = self._hf_model(**inputs)
        if isinstance(response, tuple):
            return torch.stack(list(response), dim=0)
        return response


class _TwoParameterQuestionAnsweringWrapper(_QuestionAnsweringWrapperModule):
    def __init__(self, model: PreTrainedModel):
        super().__init__(model=model)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Wrap the input and output to conform to the native process interface."""
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        response = self._hf_model(**inputs)
        if isinstance(response, tuple):
            return torch.stack(list(response), dim=0)
        return response


class _DistilBertWrapper(nn.Module):  # type: ignore
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
        _token_type_ids: Tensor,
        _position_ids: Tensor,
    ) -> Tensor:
        """Wrap the input and output to conform to the native process interface."""

        return self._model(input_ids=input_ids, attention_mask=attention_mask)


class _SentenceTransformerWrapperModule(nn.Module):  # type: ignore
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
        self.config = model.config

        self._remove_pooling_layer()
        self._replace_transformer_layer()

    @staticmethod
    def from_pretrained(
        model_id: str, output_key: str = DEFAULT_OUTPUT_KEY
    ) -> Optional[Any]:
        if model_id.startswith("sentence-transformers/"):
            model = AutoModel.from_pretrained(model_id, torchscript=True)
            if isinstance(
                model.config,
                (
                    transformers.MPNetConfig,
                    transformers.RobertaConfig,
                    transformers.BartConfig,
                ),
            ):
                return _TwoParameterSentenceTransformerWrapper(model, output_key)
            else:
                return _SentenceTransformerWrapper(model, output_key)
        else:
            return None

    def _remove_pooling_layer(self) -> None:
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

    def _replace_transformer_layer(self) -> None:
        """
        Replaces the HuggingFace Transformer layer in the SentenceTransformer
        modules so we can set it with one that has pooling layer removed and
        was loaded ready for TorchScript export.
        """

        self._st_model._modules["0"].auto_model = self._hf_model


class _SentenceTransformerWrapper(_SentenceTransformerWrapperModule):
    def __init__(self, model: PreTrainedModel, output_key: str = DEFAULT_OUTPUT_KEY):
        super().__init__(model=model, output_key=output_key)

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


class _TwoParameterSentenceTransformerWrapper(_SentenceTransformerWrapperModule):
    def __init__(self, model: PreTrainedModel, output_key: str = DEFAULT_OUTPUT_KEY):
        super().__init__(model=model, output_key=output_key)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Wrap the input and output to conform to the native process interface."""
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return self._st_model(inputs)[self._output_key]


class _DPREncoderWrapper(nn.Module):  # type: ignore
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
        self.config = model.config

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
        _position_ids: Tensor,
    ) -> Tensor:
        """Wrap the input and output to conform to the native process interface."""

        return self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )


class _TransformerTraceableModel(TraceableModel):
    """A base class representing a HuggingFace transformer model that can be traced."""

    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        model: Union[
            PreTrainedModel,
            _SentenceTransformerWrapperModule,
            _DPREncoderWrapper,
            _DistilBertWrapper,
        ],
    ):
        super(_TransformerTraceableModel, self).__init__(model=model)
        self._tokenizer = tokenizer

    def _trace(self) -> TracedModelTypes:
        inputs = self._prepare_inputs()

        # Add params when not provided by the tokenizer (e.g. DistilBERT), to conform to BERT interface
        if "token_type_ids" not in inputs:
            inputs["token_type_ids"] = torch.zeros(
                inputs["input_ids"].size(1), dtype=torch.long
            )
        if isinstance(
            self._model.config,
            (
                transformers.MPNetConfig,
                transformers.RobertaConfig,
                transformers.BartConfig,
            ),
        ):
            return torch.jit.trace(
                self._model,
                (inputs["input_ids"], inputs["attention_mask"]),
            )

        position_ids = torch.arange(inputs["input_ids"].size(1), dtype=torch.long)

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


class _TraceableClassificationModel(_TransformerTraceableModel, ABC):
    def classification_labels(self) -> Optional[List[str]]:
        id_label_items = self._model.config.id2label.items()
        labels = [v for _, v in sorted(id_label_items, key=lambda kv: kv[0])]  # type: ignore

        # Make classes like I-PER into I_PER which fits Java enumerations
        return [label.replace("-", "_") for label in labels]


class _TraceableFillMaskModel(_TransformerTraceableModel):
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


class _TraceableTextEmbeddingModel(_TransformerTraceableModel):
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


class _TraceableQuestionAnsweringModel(_TransformerTraceableModel):
    def _prepare_inputs(self) -> transformers.BatchEncoding:
        return self._tokenizer(
            "What is the meaning of life?"
            "The meaning of life, according to the hitchikers guide, is 42.",
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
        self._vocab = self._load_vocab()
        self._config = self._create_config()

    def _load_vocab(self) -> Dict[str, List[str]]:
        vocab_items = self._tokenizer.get_vocab().items()
        vocabulary = [k for k, _ in sorted(vocab_items, key=lambda kv: kv[1])]  # type: ignore
        vocab_obj = {
            "vocabulary": vocabulary,
        }
        ranks = getattr(self._tokenizer, "bpe_ranks", {})
        if len(ranks) > 0:
            merges = [
                " ".join(m) for m, _ in sorted(ranks.items(), key=lambda kv: kv[1])
            ]
            vocab_obj["merges"] = merges
        return vocab_obj

    def _create_tokenization_config(self) -> NlpTokenizationConfig:
        if isinstance(self._tokenizer, transformers.MPNetTokenizer):
            return NlpMPNetTokenizationConfig(
                do_lower_case=getattr(self._tokenizer, "do_lower_case", None),
                max_sequence_length=getattr(
                    self._tokenizer, "max_model_input_sizes", dict()
                ).get(self._model_id),
            )
        elif isinstance(
            self._tokenizer, (transformers.RobertaTokenizer, transformers.BartTokenizer)
        ):
            return NlpRobertaTokenizationConfig(
                add_prefix_space=getattr(self._tokenizer, "add_prefix_space", None),
                max_sequence_length=getattr(
                    self._tokenizer, "max_model_input_sizes", dict()
                ).get(self._model_id),
            )
        else:
            return NlpBertTokenizationConfig(
                do_lower_case=getattr(self._tokenizer, "do_lower_case", None),
                max_sequence_length=getattr(
                    self._tokenizer, "max_model_input_sizes", dict()
                ).get(self._model_id),
            )

    def _create_config(self) -> NlpTrainedModelConfig:
        tokenization_config = self._create_tokenization_config()

        # Set squad well known defaults
        if self._task_type == "question_answering":
            tokenization_config.max_sequence_length = 386
            tokenization_config.span = 128
            tokenization_config.truncate = "none"
        inference_config = (
            TASK_TYPE_TO_INFERENCE_CONFIG[self._task_type](
                tokenization=tokenization_config,
                classification_labels=self._traceable_model.classification_labels(),
            )
            if self._traceable_model.classification_labels()
            else TASK_TYPE_TO_INFERENCE_CONFIG[self._task_type](
                tokenization=tokenization_config
            )
        )

        return NlpTrainedModelConfig(
            description=f"Model {self._model_id} for task type '{self._task_type}'",
            model_type="pytorch",
            inference_config=inference_config,
            input=TrainedModelInput(
                field_names=["text_field"],
            ),
        )

    def _create_traceable_model(self) -> TraceableModel:
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
            model = _SentenceTransformerWrapperModule.from_pretrained(self._model_id)
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
        elif self._task_type == "question_answering":
            model = _QuestionAnsweringWrapperModule.from_pretrained(self._model_id)
            return _TraceableQuestionAnsweringModel(self._tokenizer, model)
        else:
            raise TypeError(
                f"Unknown task type {self._task_type}, must be one of: {SUPPORTED_TASK_TYPES_NAMES}"
            )

    def elasticsearch_model_id(self) -> str:
        # Elasticsearch model IDs need to be a specific format: no special chars, all lowercase, max 64 chars
        return self._model_id.replace("/", "__").lower()[:64]

    def save(self, path: str) -> Tuple[str, NlpTrainedModelConfig, str]:
        # save traced model
        model_path = self._traceable_model.save(path)

        # save vocabulary
        vocab_path = os.path.join(path, "vocabulary.json")
        with open(vocab_path, "w") as outfile:
            json.dump(self._vocab, outfile)

        return model_path, self._config, vocab_path
