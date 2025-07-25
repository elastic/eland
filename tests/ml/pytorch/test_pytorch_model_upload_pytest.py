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
import platform
import tempfile

import pytest
from elasticsearch import NotFoundError

try:
    import sklearn  # noqa: F401

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from eland.ml.pytorch import PyTorchModel
    from eland.ml.pytorch.transformers import TransformerModel

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from tests import ES_TEST_CLIENT, ES_VERSION

pytestmark = [
    pytest.mark.skipif(
        not HAS_SKLEARN, reason="This test requires 'scikit-learn' package to run"
    ),
    pytest.mark.skipif(
        not HAS_TRANSFORMERS, reason="This test requires 'transformers' package to run"
    ),
]

TEXT_PREDICTION_MODELS = [
    (
        "distilbert-base-uncased",
        "fill_mask",
        "[MASK] is the capital of France.",
        "paris",
    )
]

TEXT_EMBEDDING_MODELS = [
    (
        "sentence-transformers/all-MiniLM-L6-v2",
        "text_embedding",
        "Paris is the capital of France.",
    )
]

TEXT_SIMILARITY_MODELS = ["mixedbread-ai/mxbai-rerank-xsmall-v1"]

TEXT_EXPANSION_MODELS = ["naver/splade-v3-distilbert"]


@pytest.fixture(scope="function", autouse=True)
def setup_and_tear_down():
    ES_TEST_CLIENT.cluster.put_settings(
        body={"transient": {"logger.org.elasticsearch.xpack.ml": "DEBUG"}}
    )
    yield
    for model_id, _, _, _ in TEXT_PREDICTION_MODELS:
        model = PyTorchModel(ES_TEST_CLIENT, model_id.replace("/", "__").lower()[:64])
        try:
            model.stop()
            model.delete()
        except NotFoundError:
            pass


@pytest.fixture(scope="session")
def quantize():
    # quantization does not work on ARM processors
    # TODO: It seems that PyTorch 2.0 supports OneDNN for aarch64. We should
    # revisit this when we upgrade to PyTorch 2.0.
    return platform.machine() not in ["arm64", "aarch64"]


def download_model_and_start_deployment(tmp_dir, quantize, model_id, task):
    print("Loading HuggingFace transformer tokenizer and model")
    tm = TransformerModel(
        model_id=model_id, task_type=task, es_version=ES_VERSION, quantize=quantize
    )
    model_path, config, vocab_path = tm.save(tmp_dir)
    ptm = PyTorchModel(ES_TEST_CLIENT, tm.elasticsearch_model_id())
    try:
        ptm.stop()
        ptm.delete()
    except NotFoundError:
        pass
    print(f"Importing model: {ptm.model_id}")
    ptm.import_model(
        model_path=model_path, config_path=None, vocab_path=vocab_path, config=config
    )
    ptm.start()
    return ptm


class TestPytorchModel:
    @pytest.mark.parametrize("model_id,task,text_input,value", TEXT_PREDICTION_MODELS)
    def test_text_prediction(self, model_id, task, text_input, value, quantize):
        with tempfile.TemporaryDirectory() as tmp_dir:
            ptm = download_model_and_start_deployment(tmp_dir, quantize, model_id, task)
            results = ptm.infer(docs=[{"text_field": text_input}])
            assert results.body["inference_results"][0]["predicted_value"] == value

    @pytest.mark.parametrize("model_id,task,text_input", TEXT_EMBEDDING_MODELS)
    def test_text_embedding(self, model_id, task, text_input, quantize):
        with tempfile.TemporaryDirectory() as tmp_dir:
            ptm = download_model_and_start_deployment(tmp_dir, quantize, model_id, task)
            ptm.infer(docs=[{"text_field": text_input}])

            if ES_VERSION >= (8, 8, 0):
                configs = ES_TEST_CLIENT.ml.get_trained_models(model_id=ptm.model_id)
                assert (
                    int(
                        configs["trained_model_configs"][0]["inference_config"][
                            "text_embedding"
                        ]["embedding_size"]
                    )
                    > 0
                )

    @pytest.mark.skipif(
        ES_VERSION < (8, 16, 0), reason="requires 8.16.0 for DeBERTa models"
    )
    @pytest.mark.parametrize("model_id", TEXT_SIMILARITY_MODELS)
    def test_text_similarity(self, model_id):
        with tempfile.TemporaryDirectory() as tmp_dir:
            ptm = download_model_and_start_deployment(
                tmp_dir, False, model_id, "text_similarity"
            )
            result = ptm.infer(
                docs=[
                    {
                        "text_field": "The Amazon rainforest covers most of the Amazon basin in South America"
                    },
                    {"text_field": "Paris is the capital of France"},
                ],
                inference_config={"text_similarity": {"text": "France"}},
            )

            assert result.body["inference_results"][0]["predicted_value"] < 0
            assert result.body["inference_results"][1]["predicted_value"] > 0

    @pytest.mark.skipif(ES_VERSION < (9, 0, 0), reason="requires current major version")
    @pytest.mark.parametrize("model_id", TEXT_EXPANSION_MODELS)
    def test_text_expansion(self, model_id):
        with tempfile.TemporaryDirectory() as tmp_dir:
            ptm = download_model_and_start_deployment(
                tmp_dir, False, model_id, "text_expansion"
            )
            result = ptm.infer(
                docs=[
                    {
                        "text_field": "The Amazon rainforest covers most of the Amazon basin in South America"
                    },
                    {"text_field": "Paris is the capital of France"},
                ]
            )

            assert len(result.body["inference_results"][0]["predicted_value"]) > 0
            assert len(result.body["inference_results"][1]["predicted_value"]) > 0
