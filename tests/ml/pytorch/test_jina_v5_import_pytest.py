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
Tests for importing jinaai/jina-embeddings-v5-text-nano into Elasticsearch
via eland. Validates tokenizer loading, model tracing, and embedding fidelity.
"""

import json
import os
import tempfile

import numpy as np
import pytest

try:
    import torch

    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False

pytestmark = [
    pytest.mark.skipif(
        not HAS_PYTORCH, reason="This test requires 'pytorch' package to run"
    ),
    pytest.mark.skipif(
        not os.environ.get("TEST_JINA_INTEGRATION"),
        reason="Set TEST_JINA_INTEGRATION=1 to run Jina v5 integration tests "
        "(downloads ~424 MB model from HuggingFace)",
    ),
]

MODEL_ID = "jinaai/jina-embeddings-v5-text-nano"

SAMPLE_TEXTS = [
    "Elasticsearch is a distributed search engine.",
    "The quick brown fox jumps over the lazy dog.",
]


@pytest.fixture(scope="module")
def transformer_model():
    from eland.ml.pytorch.transformers import TransformerModel

    return TransformerModel(
        model_id=MODEL_ID,
        task_type="text_embedding",
        trust_remote_code=True,
        es_version=(8, 8, 0),
    )


@pytest.fixture(scope="module")
def saved_model(transformer_model):
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path, config, vocab_path = transformer_model.save(tmp_dir)

        with open(vocab_path) as f:
            vocab = json.load(f)

        yield {
            "model_path": model_path,
            "config": config,
            "vocab": vocab,
            "tmp_dir": tmp_dir,
        }


class TestJinaV5TokenizerLoading:
    def test_tokenizer_is_fast(self, transformer_model):
        from transformers import PreTrainedTokenizerFast

        assert isinstance(transformer_model._tokenizer, PreTrainedTokenizerFast)

    def test_model_detected_as_jina_v5(self, transformer_model):
        assert transformer_model._is_jina_v5()


class TestJinaV5VocabExtraction:
    def test_vocab_has_vocabulary(self, saved_model):
        vocab = saved_model["vocab"]
        assert "vocabulary" in vocab
        assert len(vocab["vocabulary"]) == 128256

    def test_vocab_has_merges(self, saved_model):
        vocab = saved_model["vocab"]
        assert "merges" in vocab
        assert len(vocab["merges"]) > 0

    def test_special_tokens_in_vocab(self, saved_model):
        vocabulary = saved_model["vocab"]["vocabulary"]
        assert "<|begin_of_text|>" in vocabulary
        assert "<|end_of_text|>" in vocabulary
        assert "<|mask|>" in vocabulary


class TestJinaV5Config:
    def test_config_tokenization_type(self, saved_model):
        config = saved_model["config"]
        config_dict = config.to_dict()
        inference_config = config_dict["inference_config"]
        text_embedding = inference_config["text_embedding"]
        tokenization = text_embedding["tokenization"]
        assert "roberta" in tokenization

    def test_config_has_embedding_size(self, saved_model):
        config = saved_model["config"]
        config_dict = config.to_dict()
        inference_config = config_dict["inference_config"]
        text_embedding = inference_config["text_embedding"]
        assert "embedding_size" in text_embedding
        assert text_embedding["embedding_size"] == 768


class TestJinaV5Tracing:
    def test_traced_model_loads(self, saved_model):
        model_path = saved_model["model_path"]
        assert os.path.exists(model_path)
        traced = torch.jit.load(model_path)
        assert traced is not None

    def test_traced_model_produces_output(self, saved_model, transformer_model):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        inputs = tokenizer(
            SAMPLE_TEXTS,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        traced = torch.jit.load(saved_model["model_path"])
        with torch.no_grad():
            output = traced(inputs["input_ids"], inputs["attention_mask"])

        assert output.shape[0] == len(SAMPLE_TEXTS)
        assert output.shape[1] == 768

        # Verify L2 normalization
        norms = torch.norm(output, p=2, dim=-1)
        np.testing.assert_allclose(norms.numpy(), 1.0, atol=1e-5)


class TestJinaV5TracedVsHuggingFace:
    """Validate that the traced model produces the same embeddings as the
    original HuggingFace model."""

    def test_embeddings_match(self, saved_model):
        from transformers import AutoModel, AutoTokenizer

        # 1. Get reference embeddings from the original HF model
        hf_model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
        hf_model.eval()
        with torch.no_grad():
            hf_embeddings = hf_model.encode(
                SAMPLE_TEXTS,
                task="retrieval",
                prompt_name="document",
            )

        # 2. Prepare inputs for the traced model (must manually add prefix)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        prefixed_texts = [f"Document: {t}" for t in SAMPLE_TEXTS]
        inputs = tokenizer(
            prefixed_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        # 3. Run through traced model
        traced = torch.jit.load(saved_model["model_path"])
        with torch.no_grad():
            traced_embeddings = traced(inputs["input_ids"], inputs["attention_mask"])

        # 4. Compare
        np.testing.assert_allclose(
            hf_embeddings.detach().cpu().numpy(),
            traced_embeddings.detach().cpu().numpy(),
            rtol=1e-4,
            atol=1e-5,
        )
