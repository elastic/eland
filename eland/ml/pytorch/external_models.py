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

import json
import os.path
import torch
import transformers
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
from torch import nn, Tensor
from transformers import AutoConfig, AutoModel, PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import Any, Dict, List, Optional, Tuple, Union

DEFAULT_OUTPUT_KEY = 'sentence_embedding'
SUPPORTED_TASK_TYPES = {'fill_mask', 'ner', 'text_classification', 'text_embedding'}
SUPPORTED_TASK_TYPES_STR = ', '.join(SUPPORTED_TASK_TYPES)
SUPPORTED_TOKENIZERS = (
    transformers.BertTokenizer,
    transformers.BertTokenizerFast,
    transformers.DPRContextEncoderTokenizer,
    transformers.DPRContextEncoderTokenizerFast,
    transformers.DPRQuestionEncoderTokenizer,
    transformers.DPRQuestionEncoderTokenizerFast,
    transformers.DistilBertTokenizer,
    transformers.DistilBertTokenizerFast,
    transformers.ElectraTokenizer,
    transformers.ElectraTokenizerFast,
    transformers.MobileBertTokenizer,
    transformers.MobileBertTokenizerFast,
    transformers.RetriBertTokenizer,
    transformers.RetriBertTokenizerFast,
)
SUPPORTED_TOKENIZERS_STR = ', '.join([str(x) for x in SUPPORTED_TOKENIZERS])

TracedModelTypes = Union[torch.nn.Module, torch.ScriptModule, torch.jit.ScriptModule, torch.jit.TopLevelTracedModule]


class DistilBertWrapper(nn.Module):
    def __init__(self, model: transformers.DistilBertModel):
        super(DistilBertWrapper, self).__init__()
        self._model = model
        self.config = model.config

    @staticmethod
    def try_wrapping(model: PreTrainedModel) -> Optional[Any]:
        if isinstance(model.config, transformers.DistilBertConfig):
            return DistilBertWrapper(model)
        else:
            return model

    def forward(self, input_ids: Tensor, attention_mask: Tensor,
                token_type_ids: Tensor, position_ids: Tensor) -> Tensor:
        """Wrap the input and output to conform to the native process interface."""

        return self._model(input_ids=input_ids, attention_mask=attention_mask)


class SentenceTransformerWrapper(nn.Module):
    def __init__(self, model: PreTrainedModel, output_key: str = DEFAULT_OUTPUT_KEY):
        super(SentenceTransformerWrapper, self).__init__()
        self._hf_model = model
        self._st_model = SentenceTransformer(model.name_or_path)
        self._output_key = output_key

        self._remove_pooling_layer()
        self._replace_transformer_layer()

    @staticmethod
    def from_pretrained(model_id: str, output_key: str = DEFAULT_OUTPUT_KEY) -> Optional[Any]:
        if model_id.startswith('sentence-transformers/'):
            model = AutoModel.from_pretrained(model_id, torchscript=True)
            return SentenceTransformerWrapper(model, output_key)
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

        if getattr(self._hf_model, 'pooler', lambda: None):
            self._hf_model.pooler = None

    def _replace_transformer_layer(self):
        """
        Replaces the HuggingFace Transformer layer in the SentenceTransformer
        modules so we can set it with one that has pooling layer removed and
        was loaded ready for TorchScript export.
        """

        self._st_model._modules['0'].auto_model = self._hf_model

    def forward(self, input_ids: Tensor, attention_mask: Tensor,
                token_type_ids: Tensor, position_ids: Tensor) -> Tensor:
        """Wrap the input and output to conform to the native process interface."""

        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'position_ids': position_ids,
        }

        # remove inputs for specific model types
        if isinstance(self._hf_model.config, transformers.DistilBertConfig):
            del inputs['token_type_ids']

        return self._st_model(inputs)[self._output_key]


class DPREncoderWrapper(nn.Module):
    """
    AutoModel loading does not work for DPRContextEncoders, this only exists as
    a workaround.
    See: https://github.com/huggingface/transformers/issues/13670
    """
    _SUPPORTED_MODELS = {
        transformers.DPRContextEncoder,
        transformers.DPRQuestionEncoder,
    }
    _SUPPORTED_MODELS_STR = set([x.__name__ for x in _SUPPORTED_MODELS])

    def __init__(self, model: Union[transformers.DPRContextEncoder, transformers.DPRQuestionEncoder]):
        super(DPREncoderWrapper, self).__init__()
        self._model = model

    @staticmethod
    def from_pretrained(model_id: str) -> Optional[Any]:

        config = AutoConfig.from_pretrained(model_id)

        def is_compatible() -> bool:
            is_dpr_model = config.model_type == 'dpr'
            has_architectures = len(config.architectures) == 1
            is_supported_architecture = config.architectures[0] in DPREncoderWrapper._SUPPORTED_MODELS_STR
            return is_dpr_model and has_architectures and is_supported_architecture

        if is_compatible():
            model = getattr(transformers, config.architectures[0]).from_pretrained(model_id, torchscript=True)
            return DPREncoderWrapper(model)
        else:
            return None

    def forward(self, input_ids: Tensor, attention_mask: Tensor,
                token_type_ids: Tensor, position_ids: Tensor) -> Tensor:
        """Wrap the input and output to conform to the native process interface."""

        return self._model(**{
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
        })


class HuggingFaceTransformerModel:

    class TraceableModel(ABC):
        def __init__(
                self,
                tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
                model: Union[PreTrainedModel, SentenceTransformerWrapper, DPREncoderWrapper, DistilBertWrapper]
        ):
            self._tokenizer = tokenizer
            self._model = model

        def classification_labels(self) -> Optional[List[str]]:
            return None

        @abstractmethod
        def trace(self) -> TracedModelTypes:
            ...

    class TraceableClassificationModel(TraceableModel, ABC):
        def classification_labels(self) -> Optional[List[str]]:
            labels = HuggingFaceTransformerModel._dict_to_ordered_list(self._model.config.id2label, sort_by_key=True)
            # Make classes like I-PER into I_PER which fits Java enumerations
            return [x.replace('-', '_') for x in labels]

    class FillMaskModel(TraceableModel):
        def trace(self) -> TracedModelTypes:
            # model needs to be in evaluate mode
            self._model.eval()

            # tokenizing dummy input text
            text = "[CLS] Who was Jim Henson ? [SEP] [MASK] Henson was a puppeteer [SEP]"
            tokenized_text = self._tokenizer.tokenize(text)

            # create token and segment IDs
            indexed_tokens = self._tokenizer.convert_tokens_to_ids(tokenized_text)
            segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

            # prepare dummy input tensors
            tokens_tensor = torch.tensor([indexed_tokens])
            attention_mask = torch.ones(tokens_tensor.size())
            segments_tensors = torch.tensor([segments_ids])
            position_ids = torch.arange(tokens_tensor.size(1))

            return torch.jit.trace(self._model, (tokens_tensor, attention_mask, segments_tensors, position_ids))

    class NerModel(TraceableClassificationModel):
        def trace(self) -> TracedModelTypes:
            # model needs to be in evaluate mode
            self._model.eval()

            # tokenizing dummy input text
            text = "Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO, therefore very" \
                   "close to the Manhattan Bridge."
            tokenized_text = self._tokenizer.tokenize(text)

            # create token IDs
            indexed_tokens = self._tokenizer.convert_tokens_to_ids(tokenized_text)

            # prepare dummy input tensors
            tokens_tensor = torch.tensor([indexed_tokens])
            attention_mask = torch.ones(tokens_tensor.size())
            token_type_ids = torch.zeros(tokens_tensor.size(), dtype=torch.long)
            position_ids = torch.arange(tokens_tensor.size(1), dtype=torch.long)

            return torch.jit.trace(self._model, (tokens_tensor, attention_mask, token_type_ids, position_ids))

    class TextClassificationModel(TraceableClassificationModel):
        def trace(self) -> TracedModelTypes:
            # model needs to be in evaluate mode
            self._model.eval()

            # tokenizing dummy input text
            text = "The cat was sick on the bed."
            tokenized_text = self._tokenizer.tokenize(text)

            # create token IDs
            indexed_tokens = self._tokenizer.convert_tokens_to_ids(tokenized_text)

            # prepare dummy input tensors
            tokens_tensor = torch.tensor([indexed_tokens])
            attention_mask = torch.ones(tokens_tensor.size())
            token_type_ids = torch.zeros(tokens_tensor.size(), dtype=torch.long)
            position_ids = torch.arange(tokens_tensor.size(1), dtype=torch.long)

            return torch.jit.trace(self._model, (tokens_tensor, attention_mask, token_type_ids, position_ids))

    class TextEmbeddingModel(TraceableModel):
        def trace(self) -> TracedModelTypes:
            # model needs to be in evaluate mode
            self._model.eval()

            # tokenizing dummy input text
            text = "This is an example sentence"
            tokenized_text = self._tokenizer.tokenize(text)

            # create token IDs
            indexed_tokens = self._tokenizer.convert_tokens_to_ids(tokenized_text)

            # prepare dummy input tensors
            tokens_tensor = torch.tensor([indexed_tokens])
            attention_mask = torch.ones(tokens_tensor.size())
            token_type_ids = torch.zeros(tokens_tensor.size(), dtype=torch.long)
            position_ids = torch.arange(tokens_tensor.size(1), dtype=torch.long)

            return torch.jit.trace(self._model, (tokens_tensor, attention_mask, token_type_ids, position_ids))

    def __init__(self, model_id: str, task_type: str):
        self._model_id = model_id
        self._task_type = task_type.replace('-', '_')

        # Elasticsearch model IDs need to be a specific format: no special chars, all lowercase, max 64 chars
        self._es_model_id = self._model_id.replace('/', '__').lower()[:64]

        # load Hugging Face model and tokenizer
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(self._model_id)

        # check for a supported tokenizer
        if not isinstance(self._tokenizer, SUPPORTED_TOKENIZERS):
            raise TypeError(f"Tokenizer type {self._tokenizer} not supported, must be one of: {SUPPORTED_TOKENIZERS_STR}")

        self._traceable_model = self._create_traceable_model()
        self._traced_model = self._traceable_model.trace()
        self._vocab = self._load_vocab()
        self._config = self._create_config()

    @staticmethod
    def _dict_to_ordered_list(dict: Dict[Any, Any], sort_by_key: bool = False, sort_by_value: bool = False) -> List[Any]:
        assert not (sort_by_key and sort_by_value)
        assert sort_by_key or sort_by_value

        if sort_by_key:
            sort_index = 0
            select_index = 1
        else:
            sort_index = 1
            select_index = 0

        l = list(dict.items())
        l.sort(key=lambda x: x[sort_index])
        new_list = [x[select_index] for x in l]
        return new_list

    def _load_vocab(self):
        vocabulary = HuggingFaceTransformerModel._dict_to_ordered_list(self._tokenizer.get_vocab(), sort_by_value=True)
        return {
            'vocabulary': vocabulary,
        }

    def _create_config(self):
        do_lower_case = hasattr(self._tokenizer, 'do_lower_case') and self._tokenizer.do_lower_case

        inference_config = {
            self._task_type: {
                'tokenization': {
                    'bert': {
                        'do_lower_case': do_lower_case,
                    }
                }
            }
        }

        if hasattr(self._tokenizer, 'max_model_input_sizes'):
            max_sequence_length = self._tokenizer.max_model_input_sizes.get(self._model_id)
            if max_sequence_length:
                inference_config[self._task_type]['tokenization']['bert']['max_sequence_length'] = max_sequence_length

        if self._traceable_model.classification_labels():
            inference_config[self._task_type]['classification_labels'] = self._traceable_model.classification_labels()

        return {
            'description': f"Model {self._model_id} for task type '{self._task_type}'",
            'model_type': 'pytorch',
            'inference_config': inference_config,
            'input': {
                'field_names': ['text_field'],
            }
        }

    def _create_traceable_model(self) -> TraceableModel:

        if self._task_type == 'fill_mask':
            model = transformers.AutoModelForMaskedLM.from_pretrained(self._model_id, torchscript=True)
            model = DistilBertWrapper.try_wrapping(model)
            return HuggingFaceTransformerModel.FillMaskModel(self._tokenizer, model)
        elif self._task_type == 'ner':
            model = transformers.AutoModelForTokenClassification.from_pretrained(self._model_id, torchscript=True)
            model = DistilBertWrapper.try_wrapping(model)
            return HuggingFaceTransformerModel.NerModel(self._tokenizer, model)
        elif self._task_type == 'text_classification':
            model = transformers.AutoModelForSequenceClassification.from_pretrained(self._model_id, torchscript=True)
            model = DistilBertWrapper.try_wrapping(model)
            return HuggingFaceTransformerModel.TextClassificationModel(self._tokenizer, model)
        elif self._task_type == 'text_embedding':
            model = SentenceTransformerWrapper.from_pretrained(self._model_id)
            if not model:
                model = DPREncoderWrapper.from_pretrained(self._model_id)
            if not model:
                model = transformers.AutoModel.from_pretrained(self._model_id, torchscript=True)
            return HuggingFaceTransformerModel.TextEmbeddingModel(self._tokenizer, model)
        else:
            raise TypeError(f"Unknown task type {self._task_type}, must be one of: {SUPPORTED_TASK_TYPES_STR}")

    def save(self, path: str) -> Tuple[str, str, str]:
        # save traced model
        model_path = os.path.join(path, 'traced_pytorch_model.pt')
        torch.jit.save(self._traced_model, model_path)

        # save configuration
        config_path = os.path.join(path, 'config.json')
        with open(config_path, 'w') as outfile:
            json.dump(self._config, outfile)

        # save vocabulary
        vocab_path = os.path.join(path, 'vocabulary.json')
        with open(vocab_path, 'w') as outfile:
            json.dump(self._vocab, outfile)

        return model_path, config_path, vocab_path
