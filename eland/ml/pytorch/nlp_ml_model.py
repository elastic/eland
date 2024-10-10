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

import typing as t


class NlpTokenizationConfig:
    def __init__(
        self,
        *,
        configuration_type: str,
        with_special_tokens: t.Optional[bool] = None,
        max_sequence_length: t.Optional[int] = None,
        truncate: t.Optional[
            t.Union["t.Literal['first', 'none', 'second']", str]
        ] = None,
        span: t.Optional[int] = None,
    ):
        self.name = configuration_type
        self.with_special_tokens = with_special_tokens
        self.max_sequence_length = max_sequence_length
        self.truncate = truncate
        self.span = span

    def to_dict(self):
        return {
            self.name: {
                k: v for k, v in self.__dict__.items() if v is not None and k != "name"
            }
        }


class NlpRobertaTokenizationConfig(NlpTokenizationConfig):
    def __init__(
        self,
        *,
        add_prefix_space: t.Optional[bool] = None,
        with_special_tokens: t.Optional[bool] = None,
        max_sequence_length: t.Optional[int] = None,
        truncate: t.Optional[
            t.Union["t.Literal['first', 'none', 'second']", str]
        ] = None,
        span: t.Optional[int] = None,
    ):
        super().__init__(
            configuration_type="roberta",
            with_special_tokens=with_special_tokens,
            max_sequence_length=max_sequence_length,
            truncate=truncate,
            span=span,
        )
        self.add_prefix_space = add_prefix_space


class NlpXLMRobertaTokenizationConfig(NlpTokenizationConfig):
    def __init__(
        self,
        *,
        with_special_tokens: t.Optional[bool] = None,
        max_sequence_length: t.Optional[int] = None,
        truncate: t.Optional[
            t.Union["t.Literal['first', 'none', 'second']", str]
        ] = None,
        span: t.Optional[int] = None,
    ):
        super().__init__(
            configuration_type="xlm_roberta",
            with_special_tokens=with_special_tokens,
            max_sequence_length=max_sequence_length,
            truncate=truncate,
            span=span,
        )


class DebertaV2Config(NlpTokenizationConfig):
    def __init__(
        self,
        *,
        do_lower_case: t.Optional[bool] = None,
        with_special_tokens: t.Optional[bool] = None,
        max_sequence_length: t.Optional[int] = None,
        truncate: t.Optional[
            t.Union["t.Literal['first', 'none', 'second']", str]
        ] = None,
        span: t.Optional[int] = None,
    ):
        super().__init__(
            configuration_type="deberta_v2",
            with_special_tokens=with_special_tokens,
            max_sequence_length=max_sequence_length,
            truncate=truncate,
            span=span,
        )


class NlpBertTokenizationConfig(NlpTokenizationConfig):
    def __init__(
        self,
        *,
        do_lower_case: t.Optional[bool] = None,
        with_special_tokens: t.Optional[bool] = None,
        max_sequence_length: t.Optional[int] = None,
        truncate: t.Optional[
            t.Union["t.Literal['first', 'none', 'second']", str]
        ] = None,
        span: t.Optional[int] = None,
    ):
        super().__init__(
            configuration_type="bert",
            with_special_tokens=with_special_tokens,
            max_sequence_length=max_sequence_length,
            truncate=truncate,
            span=span,
        )
        self.do_lower_case = do_lower_case


class NlpBertJapaneseTokenizationConfig(NlpTokenizationConfig):
    def __init__(
        self,
        *,
        do_lower_case: t.Optional[bool] = None,
        with_special_tokens: t.Optional[bool] = None,
        max_sequence_length: t.Optional[int] = None,
        truncate: t.Optional[
            t.Union["t.Literal['first', 'none', 'second']", str]
        ] = None,
        span: t.Optional[int] = None,
    ):
        super().__init__(
            configuration_type="bert_ja",
            with_special_tokens=with_special_tokens,
            max_sequence_length=max_sequence_length,
            truncate=truncate,
            span=span,
        )
        self.do_lower_case = do_lower_case


class NlpMPNetTokenizationConfig(NlpTokenizationConfig):
    def __init__(
        self,
        *,
        do_lower_case: t.Optional[bool] = None,
        with_special_tokens: t.Optional[bool] = None,
        max_sequence_length: t.Optional[int] = None,
        truncate: t.Optional[
            t.Union["t.Literal['first', 'none', 'second']", str]
        ] = None,
        span: t.Optional[int] = None,
    ):
        super().__init__(
            configuration_type="mpnet",
            with_special_tokens=with_special_tokens,
            max_sequence_length=max_sequence_length,
            truncate=truncate,
            span=span,
        )
        self.do_lower_case = do_lower_case


class InferenceConfig:
    def __init__(self, *, configuration_type: str):
        self.name = configuration_type

    def to_dict(self) -> t.Dict[str, t.Any]:
        return {
            self.name: {
                k: v.to_dict() if hasattr(v, "to_dict") else v
                for k, v in self.__dict__.items()
                if v is not None and k != "name"
            }
        }


class TextClassificationInferenceOptions(InferenceConfig):
    def __init__(
        self,
        *,
        classification_labels: t.Union[t.List[str], t.Tuple[str, ...]],
        tokenization: NlpTokenizationConfig,
        results_field: t.Optional[str] = None,
        num_top_classes: t.Optional[int] = None,
    ):
        super().__init__(configuration_type="text_classification")
        self.results_field = results_field
        self.num_top_classes = num_top_classes
        self.tokenization = tokenization
        self.classification_labels = classification_labels


class ZeroShotClassificationInferenceOptions(InferenceConfig):
    def __init__(
        self,
        *,
        tokenization: NlpTokenizationConfig,
        classification_labels: t.Union[t.List[str], t.Tuple[str, ...]],
        results_field: t.Optional[str] = None,
        multi_label: t.Optional[bool] = None,
        labels: t.Optional[t.Union[t.List[str], t.Tuple[str, ...]]] = None,
        hypothesis_template: t.Optional[str] = None,
    ):
        super().__init__(configuration_type="zero_shot_classification")
        self.tokenization = tokenization
        self.hypothesis_template = hypothesis_template
        self.classification_labels = classification_labels
        self.results_field = results_field
        self.multi_label = multi_label
        self.labels = labels


class FillMaskInferenceOptions(InferenceConfig):
    def __init__(
        self,
        *,
        tokenization: NlpTokenizationConfig,
        results_field: t.Optional[str] = None,
        num_top_classes: t.Optional[int] = None,
    ):
        super().__init__(configuration_type="fill_mask")
        self.num_top_classes = num_top_classes
        self.tokenization = tokenization
        self.results_field = results_field


class NerInferenceOptions(InferenceConfig):
    def __init__(
        self,
        *,
        tokenization: NlpTokenizationConfig,
        classification_labels: t.Union[t.List[str], t.Tuple[str, ...]],
        results_field: t.Optional[str] = None,
    ):
        super().__init__(configuration_type="ner")
        self.tokenization = tokenization
        self.classification_labels = classification_labels
        self.results_field = results_field


class PassThroughInferenceOptions(InferenceConfig):
    def __init__(
        self,
        *,
        tokenization: NlpTokenizationConfig,
        results_field: t.Optional[str] = None,
    ):
        super().__init__(configuration_type="pass_through")
        self.tokenization = tokenization
        self.results_field = results_field


class QuestionAnsweringInferenceOptions(InferenceConfig):
    def __init__(
        self,
        *,
        tokenization: NlpTokenizationConfig,
        results_field: t.Optional[str] = None,
        max_answer_length: t.Optional[int] = None,
        question: t.Optional[str] = None,
        num_top_classes: t.Optional[int] = None,
    ):
        super().__init__(configuration_type="question_answering")
        self.tokenization = tokenization
        self.results_field = results_field
        self.max_answer_length = max_answer_length
        self.question = question
        self.num_top_classes = num_top_classes


class TextSimilarityInferenceOptions(InferenceConfig):
    def __init__(
        self,
        *,
        tokenization: NlpTokenizationConfig,
        results_field: t.Optional[str] = None,
        text: t.Optional[str] = None,
    ):
        super().__init__(configuration_type="text_similarity")
        self.tokenization = tokenization
        self.results_field = results_field
        self.text = text


class TextEmbeddingInferenceOptions(InferenceConfig):
    def __init__(
        self,
        *,
        tokenization: NlpTokenizationConfig,
        results_field: t.Optional[str] = None,
        embedding_size: t.Optional[int] = None,
    ):
        super().__init__(configuration_type="text_embedding")
        self.tokenization = tokenization
        self.results_field = results_field
        self.embedding_size = embedding_size


class TextExpansionInferenceOptions(InferenceConfig):
    def __init__(
        self,
        *,
        tokenization: NlpTokenizationConfig,
        results_field: t.Optional[str] = None,
    ):
        super().__init__(configuration_type="text_expansion")
        self.tokenization = tokenization
        self.results_field = results_field


class TrainedModelInput:
    def __init__(self, *, field_names: t.List[str]):
        self.field_names = field_names

    def to_dict(self) -> t.Dict[str, t.Any]:
        return self.__dict__


class PrefixStrings:
    def __init__(
        self, *, ingest_prefix: t.Optional[str], search_prefix: t.Optional[str]
    ):
        self.ingest_prefix = ingest_prefix
        self.search_prefix = search_prefix

    def to_dict(self) -> t.Dict[str, t.Any]:
        config = {}
        if self.ingest_prefix is not None:
            config["ingest"] = self.ingest_prefix
        if self.search_prefix is not None:
            config["search"] = self.search_prefix

        return config


class NlpTrainedModelConfig:
    def __init__(
        self,
        *,
        description: str,
        inference_config: InferenceConfig,
        input: TrainedModelInput = TrainedModelInput(field_names=["text_field"]),
        metadata: t.Optional[dict] = None,
        model_type: t.Union["t.Literal['pytorch']", str] = "pytorch",
        tags: t.Optional[t.Union[t.List[str], t.Tuple[str, ...]]] = None,
        prefix_strings: t.Optional[PrefixStrings],
    ):
        self.tags = tags
        self.description = description
        self.inference_config = inference_config
        self.input = input
        self.metadata = metadata
        self.model_type = model_type
        self.prefix_strings = prefix_strings

    def to_dict(self) -> t.Dict[str, t.Any]:
        return {
            k: v.to_dict() if hasattr(v, "to_dict") else v
            for k, v in self.__dict__.items()
            if v is not None
        }
