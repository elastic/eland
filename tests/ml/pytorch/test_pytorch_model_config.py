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

import tempfile

import pytest

from eland.ml.pytorch import (
    FillMaskInferenceOptions,
    NerInferenceOptions,
    NlpBertTokenizationConfig,
    NlpMPNetTokenizationConfig,
    NlpRobertaTokenizationConfig,
    QuestionAnsweringInferenceOptions,
    TextClassificationInferenceOptions,
    TextEmbeddingInferenceOptions,
    TextSimilarityInferenceOptions,
    ZeroShotClassificationInferenceOptions,
)

try:
    import sklearn  # noqa: F401

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from eland.ml.pytorch.transformers import TransformerModel

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import torch  # noqa: F401

    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False

from tests import ES_VERSION

pytestmark = [
    pytest.mark.skipif(
        ES_VERSION < (8, 7, 0),
        reason="Eland uses Pytorch 1.13.1, versions of Elasticsearch prior to 8.7.0 are incompatible with PyTorch 1.13.1",
    ),
    pytest.mark.skipif(
        not HAS_SKLEARN, reason="This test requires 'scikit-learn' package to run"
    ),
    pytest.mark.skipif(
        not HAS_TRANSFORMERS, reason="This test requires 'transformers' package to run"
    ),
]

MODEL_CONFIGURATIONS = [
    (
        "intfloat/e5-small-v2",
        "text_embedding",
        TextEmbeddingInferenceOptions,
        NlpBertTokenizationConfig,
        512,
        384,
    ),
    (
        "sentence-transformers/all-mpnet-base-v2",
        "text_embedding",
        TextEmbeddingInferenceOptions,
        NlpMPNetTokenizationConfig,
        512,
        768,
    ),
    (
        "sentence-transformers/all-MiniLM-L12-v2",
        "text_embedding",
        TextEmbeddingInferenceOptions,
        NlpBertTokenizationConfig,
        512,
        384,
    ),
    (
        "facebook/dpr-ctx_encoder-multiset-base",
        "text_embedding",
        TextEmbeddingInferenceOptions,
        NlpBertTokenizationConfig,
        512,
        768,
    ),
    (
        "distilbert-base-uncased",
        "fill_mask",
        FillMaskInferenceOptions,
        NlpBertTokenizationConfig,
        512,
        None,
    ),
    (
        "bert-base-uncased",
        "fill_mask",
        FillMaskInferenceOptions,
        NlpBertTokenizationConfig,
        512,
        None,
    ),
    (
        "elastic/distilbert-base-uncased-finetuned-conll03-english",
        "ner",
        NerInferenceOptions,
        NlpBertTokenizationConfig,
        512,
        None,
    ),
    (
        "elastic/distilbert-base-uncased-finetuned-conll03-english",
        "ner",
        NerInferenceOptions,
        NlpBertTokenizationConfig,
        512,
        None,
    ),
    (
        "SamLowe/roberta-base-go_emotions",
        "text_classification",
        TextClassificationInferenceOptions,
        NlpRobertaTokenizationConfig,
        512,
        None,
    ),
    (
        "distilbert-base-cased-distilled-squad",
        "question_answering",
        QuestionAnsweringInferenceOptions,
        NlpBertTokenizationConfig,
        386,
        None,
    ),
    (
        "cross-encoder/ms-marco-TinyBERT-L-2-v2",
        "text_similarity",
        TextSimilarityInferenceOptions,
        NlpBertTokenizationConfig,
        512,
        None,
    ),
    (
        "valhalla/distilbart-mnli-12-6",
        "zero_shot_classification",
        ZeroShotClassificationInferenceOptions,
        NlpRobertaTokenizationConfig,
        1024,
        None,
    ),
]


class TestModelConfguration:
    @pytest.mark.parametrize(
        "model_id,task_type,config_type,tokenizer_type,max_sequence_len,embedding_size",
        MODEL_CONFIGURATIONS,
    )
    def test_text_prediction(
        self,
        model_id,
        task_type,
        config_type,
        tokenizer_type,
        max_sequence_len,
        embedding_size,
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            print("loading model " + model_id)
            tm = TransformerModel(
                model_id=model_id,
                task_type=task_type,
                es_version=ES_VERSION,
                quantize=False,
            )
            _, config, _ = tm.save(tmp_dir)
            assert "pytorch" == config.model_type
            assert ["text_field"] == config.input.field_names
            assert isinstance(config.inference_config, config_type)
            tokenization = config.inference_config.tokenization
            assert isinstance(tokenization, tokenizer_type)
            assert max_sequence_len == tokenization.max_sequence_length

            if task_type == "text_classification":
                assert isinstance(config.inference_config.classification_labels, list)
                assert len(config.inference_config.classification_labels) > 0

            if task_type == "text_embedding":
                assert embedding_size == config.inference_config.embedding_size

            if task_type == "question_answering":
                assert tokenization.truncate == "none"
                assert tokenization.span > 0

            if task_type == "zero_shot_classification":
                assert isinstance(config.inference_config.classification_labels, list)
                assert len(config.inference_config.classification_labels) > 0
