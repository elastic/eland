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
This module contains the wrapper classes for the Hugging Face models.
Wrapping is necessary to ensure that the forward method of the model
is called with the same arguments the ml-cpp pytorch_inference process
uses.
"""

from typing import Any, Optional, Union

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
)

DEFAULT_OUTPUT_KEY = "sentence_embedding"


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
    def from_pretrained(model_id: str, *, token: Optional[str] = None) -> Optional[Any]:
        model = AutoModelForQuestionAnswering.from_pretrained(
            model_id, token=token, torchscript=True
        )
        if isinstance(
            model.config,
            (
                transformers.MPNetConfig,
                transformers.XLMRobertaConfig,
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
    In Elasticsearch the BERT tokenizer is used for DistilBERT models but
    the BERT tokenizer produces 4 inputs where DistilBERT models expect 2.

    Wrap the model's forward function in a method that accepts the 4
    arguments passed to a BERT model then discard the token_type_ids
    and the position_ids to match the wrapped DistilBERT model forward
    function
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
        _token_type_ids: Tensor = None,
        _position_ids: Tensor = None,
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
        model_id: str,
        tokenizer: PreTrainedTokenizer,
        *,
        token: Optional[str] = None,
        output_key: str = DEFAULT_OUTPUT_KEY,
    ) -> Optional[Any]:
        model = AutoModel.from_pretrained(model_id, token=token, torchscript=True)
        if isinstance(
            tokenizer,
            (
                transformers.BartTokenizer,
                transformers.MPNetTokenizer,
                transformers.RobertaTokenizer,
                transformers.XLMRobertaTokenizer,
                transformers.DebertaV2Tokenizer,
            ),
        ):
            return _TwoParameterSentenceTransformerWrapper(model, output_key)
        else:
            return _SentenceTransformerWrapper(model, output_key)

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
    def from_pretrained(model_id: str, *, token: Optional[str] = None) -> Optional[Any]:
        config = AutoConfig.from_pretrained(model_id, token=token)

        def is_compatible() -> bool:
            is_dpr_model = config.model_type == "dpr"
            has_architectures = (
                config.architectures is not None and len(config.architectures) == 1
            )
            is_supported_architecture = has_architectures and (
                config.architectures[0] in _DPREncoderWrapper._SUPPORTED_MODELS_NAMES
            )
            return is_dpr_model and is_supported_architecture

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
