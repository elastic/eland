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
import random
import re
from typing import Dict, List, Optional, Set, Tuple, Union

import torch  # type: ignore
import transformers  # type: ignore
from torch import Tensor
from torch.profiler import profile  # type: ignore
from transformers import PretrainedConfig

from eland.ml.pytorch.nlp_ml_model import (
    FillMaskInferenceOptions,
    NerInferenceOptions,
    NlpTrainedModelConfig,
    PassThroughInferenceOptions,
    PrefixStrings,
    QuestionAnsweringInferenceOptions,
    TextClassificationInferenceOptions,
    TextEmbeddingInferenceOptions,
    TextExpansionInferenceOptions,
    TextSimilarityInferenceOptions,
    TrainedModelInput,
    ZeroShotClassificationInferenceOptions,
)
from eland.ml.pytorch.tokenizers import (
    SUPPORTED_TOKENIZERS,
    SUPPORTED_TOKENIZERS_NAMES,
    create_tokenization_config,
)
from eland.ml.pytorch.traceable_model import (
    TraceableFillMaskModel,
    TraceableNerModel,
    TraceablePassThroughModel,
    TraceableQuestionAnsweringModel,
    TraceableTextClassificationModel,
    TraceableTextEmbeddingModel,
    TraceableTextExpansionModel,
    TraceableTextSimilarityModel,
    TraceableZeroShotClassificationModel,
    TransformerTraceableModel,
)
from eland.ml.pytorch.wrappers import (
    _DistilBertWrapper,
    _DPREncoderWrapper,
    _QuestionAnsweringWrapperModule,
    _SentenceTransformerWrapperModule,
)

SUPPORTED_TASK_TYPES = {
    "fill_mask",
    "ner",
    "pass_through",
    "text_classification",
    "text_embedding",
    "text_expansion",
    "zero_shot_classification",
    "question_answering",
    "text_similarity",
}
ARCHITECTURE_TO_TASK_TYPE = {
    "MaskedLM": ["fill_mask", "text_embedding"],
    "TokenClassification": ["ner"],
    "SequenceClassification": [
        "text_classification",
        "zero_shot_classification",
        "text_similarity",
    ],
    "QuestionAnswering": ["question_answering"],
    "DPRQuestionEncoder": ["text_embedding"],
    "DPRContextEncoder": ["text_embedding"],
}
ZERO_SHOT_LABELS = {"contradiction", "neutral", "entailment"}
TASK_TYPE_TO_INFERENCE_CONFIG = {
    "fill_mask": FillMaskInferenceOptions,
    "ner": NerInferenceOptions,
    "text_expansion": TextExpansionInferenceOptions,
    "text_classification": TextClassificationInferenceOptions,
    "text_embedding": TextEmbeddingInferenceOptions,
    "zero_shot_classification": ZeroShotClassificationInferenceOptions,
    "pass_through": PassThroughInferenceOptions,
    "question_answering": QuestionAnsweringInferenceOptions,
    "text_similarity": TextSimilarityInferenceOptions,
}
SUPPORTED_TASK_TYPES_NAMES = ", ".join(sorted(SUPPORTED_TASK_TYPES))


TracedModelTypes = Union[
    torch.nn.Module,
    torch.ScriptModule,
    torch.jit.ScriptModule,
    torch.jit.TopLevelTracedModule,
]


class TaskTypeError(Exception):
    pass


def task_type_from_model_config(model_config: PretrainedConfig) -> Optional[str]:
    if model_config.architectures is None:
        if model_config.name_or_path.startswith("sentence-transformers/"):
            return "text_embedding"
        return None
    potential_task_types: Set[str] = set()
    for architecture in model_config.architectures:
        for substr, task_type in ARCHITECTURE_TO_TASK_TYPE.items():
            if substr in architecture:
                for t in task_type:
                    potential_task_types.add(t)
    if len(potential_task_types) == 0:
        if model_config.name_or_path.startswith("sentence-transformers/"):
            return "text_embedding"
        return None
    if (
        "text_classification" in potential_task_types
        and model_config.id2label
        and len(model_config.id2label) == 1
    ):
        return "text_similarity"
    if len(potential_task_types) > 1:
        if "zero_shot_classification" in potential_task_types:
            if model_config.label2id:
                labels = set([x.lower() for x in model_config.label2id.keys()])
                if len(labels.difference(ZERO_SHOT_LABELS)) == 0:
                    return "zero_shot_classification"
            return "text_classification"
        if "text_embedding" in potential_task_types:
            if model_config.name_or_path.startswith("sentence-transformers/"):
                return "text_embedding"
            return "fill_mask"
    return potential_task_types.pop()


class TransformerModel:
    def __init__(
        self,
        *,
        model_id: str,
        task_type: str,
        es_version: Optional[Tuple[int, int, int]] = None,
        quantize: bool = False,
        access_token: Optional[str] = None,
        ingest_prefix: Optional[str] = None,
        search_prefix: Optional[str] = None,
        max_model_input_size: Optional[int] = None,
    ):
        """
        Loads a model from the Hugging Face repository or local file and creates
        the configuration for upload to Elasticsearch.

        Parameters
        ----------
        model_id: str
            A Hugging Face model Id or a file path to the directory containing
            the model files.

        task_type: str
            One of the supported task types.

        es_version: Optional[Tuple[int, int, int]]
            The Elasticsearch cluster version.
            Certain features are created only if the target cluster is
            a high enough version to support them. If not set only
            universally supported features are added.

        quantize: bool, default False
            Quantize the model.

        access_token: Optional[str]
            For the HuggingFace Hub private model access

        ingest_prefix: Optional[str]
            Prefix string to prepend to input at ingest

        search_prefix: Optional[str]
            Prefix string to prepend to input at search

        max_model_input_size: Optional[int]
            The max model input size counted in tokens.
            Usually this value should be extracted from the model configuration
            but if that is not possible or the data is missing it can be
            explicitly set with this parameter.
        """

        self._model_id = model_id
        self._access_token = access_token
        self._task_type = task_type.replace("-", "_")
        self._ingest_prefix = ingest_prefix
        self._search_prefix = search_prefix
        self._max_model_input_size = max_model_input_size

        # load Hugging Face model and tokenizer
        # use padding in the tokenizer to ensure max length sequences are used for tracing (at call time)
        #  - see: https://huggingface.co/transformers/serialization.html#dummy-inputs-and-standard-lengths
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            self._model_id,
            token=self._access_token,
            use_fast=True,  # TODO not all tokenizers support fast mode
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
        self._config = self._create_config(es_version)

    def _load_vocab(self) -> Dict[str, List[str]]:
        vocab_items = self._tokenizer.get_vocab().items()
        vocabulary = [k for k, _ in sorted(vocab_items, key=lambda kv: kv[1])]
        vocab_obj = {
            "vocabulary": vocabulary,
        }
        ranks = getattr(self._tokenizer, "bpe_ranks", {})
        if len(ranks) > 0:
            merges = [
                " ".join(m) for m, _ in sorted(ranks.items(), key=lambda kv: kv[1])
            ]
            vocab_obj["merges"] = merges

        if isinstance(self._tokenizer, transformers.DebertaV2Tokenizer):
            sp_model = self._tokenizer._tokenizer.spm
        else:
            sp_model = getattr(self._tokenizer, "sp_model", None)

        if sp_model:
            id_correction = getattr(self._tokenizer, "fairseq_offset", 0)
            scores = []
            for _ in range(0, id_correction):
                scores.append(0.0)
            for token_id in range(id_correction, len(vocabulary)):
                try:
                    scores.append(sp_model.get_score(token_id - id_correction))
                except IndexError:
                    scores.append(0.0)
                    pass
            if len(scores) > 0:
                vocab_obj["scores"] = scores
        return vocab_obj

    def _create_config(
        self, es_version: Optional[Tuple[int, int, int]]
    ) -> NlpTrainedModelConfig:
        tokenization_config = create_tokenization_config(
            self._model_id, self._max_model_input_size, self._tokenizer
        )

        # Set squad well known defaults
        if self._task_type == "question_answering":
            tokenization_config.max_sequence_length = 386
            tokenization_config.span = 128
            tokenization_config.truncate = "none"

        if self._task_type == "text_similarity":
            tokenization_config.truncate = "second"

        if self._traceable_model.classification_labels():
            inference_config = TASK_TYPE_TO_INFERENCE_CONFIG[self._task_type](
                tokenization=tokenization_config,
                classification_labels=self._traceable_model.classification_labels(),
            )
        elif self._task_type == "text_embedding":
            # The embedding_size paramater was added in Elasticsearch 8.8
            # If the version is not known use the basic config
            if es_version is None or (es_version[0] <= 8 and es_version[1] < 8):
                inference_config = TASK_TYPE_TO_INFERENCE_CONFIG[self._task_type](
                    tokenization=tokenization_config
                )
            else:
                sample_embedding = self._traceable_model.sample_output()
                if type(sample_embedding) is tuple:
                    text_embedding = sample_embedding[0]
                else:
                    text_embedding = sample_embedding

                embedding_size = text_embedding.size(-1)
                inference_config = TASK_TYPE_TO_INFERENCE_CONFIG[self._task_type](
                    tokenization=tokenization_config,
                    embedding_size=embedding_size,
                )
        else:
            inference_config = TASK_TYPE_TO_INFERENCE_CONFIG[self._task_type](
                tokenization=tokenization_config
            )

        # add static and dynamic memory state size to metadata
        per_deployment_memory_bytes = self._get_per_deployment_memory()

        per_allocation_memory_bytes = self._get_per_allocation_memory(
            tokenization_config.max_sequence_length, 1
        )

        metadata = {
            "per_deployment_memory_bytes": per_deployment_memory_bytes,
            "per_allocation_memory_bytes": per_allocation_memory_bytes,
        }

        prefix_strings = (
            PrefixStrings(
                ingest_prefix=self._ingest_prefix, search_prefix=self._search_prefix
            )
            if self._ingest_prefix or self._search_prefix
            else None
        )
        prefix_strings_supported = es_version is None or es_version >= (8, 12, 0)
        if not prefix_strings_supported and prefix_strings:
            raise Exception(
                f"The Elasticsearch cluster version {es_version} does not support prefix strings. Support was added in version 8.12.0"
            )

        return NlpTrainedModelConfig(
            description=f"Model {self._model_id} for task type '{self._task_type}'",
            model_type="pytorch",
            inference_config=inference_config,
            input=TrainedModelInput(
                field_names=["text_field"],
            ),
            metadata=metadata,
            prefix_strings=prefix_strings,
        )

    def _get_per_deployment_memory(self) -> float:
        """
        Returns the static memory size of the model in bytes.
        """
        psize: float = sum(
            param.nelement() * param.element_size()
            for param in self._traceable_model.model.parameters()
        )
        bsize: float = sum(
            buffer.nelement() * buffer.element_size()
            for buffer in self._traceable_model.model.buffers()
        )
        return psize + bsize

    def _get_per_allocation_memory(
        self, max_seq_length: Optional[int], batch_size: int
    ) -> float:
        """
        Returns the transient memory size of the model in bytes.

        Parameters
        ----------
        max_seq_length : Optional[int]
            Maximum sequence length to use for the model.
        batch_size : int
            Batch size to use for the model.
        """
        activities = [torch.profiler.ProfilerActivity.CPU]

        # Get the memory usage of the model with a batch size of 1.
        inputs_1 = self._get_model_inputs(max_seq_length, 1)
        with profile(activities=activities, profile_memory=True) as prof:
            self._traceable_model.model(*inputs_1)
        mem1: float = prof.key_averages().total_average().cpu_memory_usage

        # This is measuring memory usage of the model with a batch size of 2 and
        # then linearly extrapolating it to get the memory usage of the model for
        # a batch size of batch_size.
        if batch_size == 1:
            return mem1
        inputs_2 = self._get_model_inputs(max_seq_length, 2)
        with profile(activities=activities, profile_memory=True) as prof:
            self._traceable_model.model(*inputs_2)
        mem2: float = prof.key_averages().total_average().cpu_memory_usage
        return mem1 + (mem2 - mem1) * (batch_size - 1)

    def _get_model_inputs(
        self,
        max_length: Optional[int],
        batch_size: int,
    ) -> Tuple[Tensor, ...]:
        """
        Returns a random batch of inputs for the model.

        Parameters
        ----------
        max_length : Optional[int]
            Maximum sequence length to use for the model. Default is 512.
        batch_size : int
            Batch size to use for the model.
        """
        vocab: list[str] = list(self._tokenizer.get_vocab().keys())

        # if optional max_length is not set, set it to 512
        if max_length is None:
            max_length = 512

        # generate random text
        texts: list[str] = [
            " ".join(random.choices(vocab, k=max_length)) for _ in range(batch_size)
        ]

        # tokenize text
        inputs: transformers.BatchEncoding = self._tokenizer(
            texts,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )

        return self._make_inputs_compatible(inputs)

    def _make_inputs_compatible(
        self, inputs: transformers.BatchEncoding
    ) -> Tuple[Tensor, ...]:
        """ "
        Make the input batch format compatible to the model's requirements.

        Parameters
        ----------
        inputs : transformers.BatchEncoding
            The input batch to make compatible.
        """
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
            del inputs["token_type_ids"]
            return (inputs["input_ids"], inputs["attention_mask"])

        position_ids = torch.arange(inputs["input_ids"].size(1), dtype=torch.long)
        inputs["position_ids"] = position_ids
        return (
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["token_type_ids"],
            inputs["position_ids"],
        )

    def _create_traceable_model(self) -> TransformerTraceableModel:
        if self._task_type == "auto":
            model = transformers.AutoModel.from_pretrained(
                self._model_id, token=self._access_token, torchscript=True
            )
            maybe_task_type = task_type_from_model_config(model.config)
            if maybe_task_type is None:
                raise TaskTypeError(
                    f"Unable to automatically determine task type for model {self._model_id}, please supply task type: {SUPPORTED_TASK_TYPES_NAMES}"
                )
            else:
                self._task_type = maybe_task_type

        if self._task_type == "text_expansion":
            model = transformers.AutoModelForMaskedLM.from_pretrained(
                self._model_id, token=self._access_token, torchscript=True
            )
            model = _DistilBertWrapper.try_wrapping(model)
            return TraceableTextExpansionModel(self._tokenizer, model)

        if self._task_type == "fill_mask":
            model = transformers.AutoModelForMaskedLM.from_pretrained(
                self._model_id, token=self._access_token, torchscript=True
            )
            model = _DistilBertWrapper.try_wrapping(model)
            return TraceableFillMaskModel(self._tokenizer, model)

        elif self._task_type == "ner":
            model = transformers.AutoModelForTokenClassification.from_pretrained(
                self._model_id, token=self._access_token, torchscript=True
            )
            model = _DistilBertWrapper.try_wrapping(model)
            return TraceableNerModel(self._tokenizer, model)

        elif self._task_type == "text_classification":
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                self._model_id, token=self._access_token, torchscript=True
            )
            model = _DistilBertWrapper.try_wrapping(model)
            return TraceableTextClassificationModel(self._tokenizer, model)

        elif self._task_type == "text_embedding":
            model = _DPREncoderWrapper.from_pretrained(
                self._model_id, token=self._access_token
            )
            if not model:
                model = _SentenceTransformerWrapperModule.from_pretrained(
                    self._model_id, self._tokenizer, token=self._access_token
                )
            return TraceableTextEmbeddingModel(self._tokenizer, model)

        elif self._task_type == "zero_shot_classification":
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                self._model_id, token=self._access_token, torchscript=True
            )
            model = _DistilBertWrapper.try_wrapping(model)
            return TraceableZeroShotClassificationModel(self._tokenizer, model)

        elif self._task_type == "question_answering":
            model = _QuestionAnsweringWrapperModule.from_pretrained(
                self._model_id, token=self._access_token
            )
            return TraceableQuestionAnsweringModel(self._tokenizer, model)

        elif self._task_type == "text_similarity":
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                self._model_id, token=self._access_token, torchscript=True
            )
            model = _DistilBertWrapper.try_wrapping(model)
            return TraceableTextSimilarityModel(self._tokenizer, model)

        elif self._task_type == "pass_through":
            model = transformers.AutoModel.from_pretrained(
                self._model_id, token=self._access_token, torchscript=True
            )
            return TraceablePassThroughModel(self._tokenizer, model)

        else:
            raise TypeError(
                f"Task {self._task_type} is not supported, must be one of: {SUPPORTED_TASK_TYPES_NAMES}"
            )

    def elasticsearch_model_id(self) -> str:
        return elasticsearch_model_id(self._model_id)

    def save(self, path: str) -> Tuple[str, NlpTrainedModelConfig, str]:
        # save traced model
        model_path = self._traceable_model.save(path)

        # save vocabulary
        vocab_path = os.path.join(path, "vocabulary.json")
        with open(vocab_path, "w") as outfile:
            json.dump(self._vocab, outfile)

        return model_path, self._config, vocab_path


def elasticsearch_model_id(model_id: str) -> str:
    """
    Elasticsearch model IDs need to be a specific format:
    no special chars, all lowercase, max 64 chars. If the
    Id is longer than 64 charaters take the last 64- in the
    case where the id is long file path this captures the
    model name.

    Ids starting with __ are not valid elasticsearch Ids,
    # this might be the case if model_id is a file path
    """

    id = re.sub(r"[\s\\/]", "__", model_id).lower()[-64:]
    id = id.removeprefix("__")
    return id
