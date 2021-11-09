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

try:
    import sklearn  # noqa: F401

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import transformers  # noqa: F401

    from eland.ml.pytorch import PyTorchModel
    from eland.ml.pytorch.transformers import TransformerModel

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from tests import ES_TEST_CLIENT, ES_VERSION

pytestmark = [
    pytest.mark.skipif(
        ES_VERSION < (8, 0, 0),
        reason="This test requires at least Elasticsearch version 8.0.0",
    ),
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


@pytest.fixture(scope="function", autouse=True)
def setup_and_tear_down():
    ES_TEST_CLIENT.cluster.put_settings(
        body={"transient": {"logger.org.elasticsearch.xpack.ml": "DEBUG"}}
    )
    yield
    for model_id, _, _, _ in TEXT_PREDICTION_MODELS:
        model = PyTorchModel(ES_TEST_CLIENT, model_id.replace("/", "__").lower()[:64])
        model.stop()
        model.delete()


def download_model_and_start_deployment(tmp_dir, quantize, model_id, task):
    print("Loading HuggingFace transformer tokenizer and model")
    tm = TransformerModel(model_id, task, quantize)
    model_path, config_path, vocab_path = tm.save(tmp_dir)
    ptm = PyTorchModel(ES_TEST_CLIENT, tm.elasticsearch_model_id())
    ptm.stop()
    ptm.delete()
    print(f"Importing model: {ptm.model_id}")
    ptm.import_model(model_path, config_path, vocab_path)
    ptm.start()
    return ptm


class TestPytorchModel:
    @pytest.mark.parametrize("model_id,task,text_input,value", TEXT_PREDICTION_MODELS)
    def test_text_classification(self, model_id, task, text_input, value):
        with tempfile.TemporaryDirectory() as tmp_dir:
            ptm = download_model_and_start_deployment(tmp_dir, True, model_id, task)
            result = ptm.infer({"docs": [{"text_field": text_input}]})
            assert result["predicted_value"] == value
