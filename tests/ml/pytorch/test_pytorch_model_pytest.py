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
    import sklearn

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import transformers

    from eland.ml.pytorch import PyTorchModel
    from eland.ml.pytorch.transformers import TransformerModel

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from tests import ES_TEST_CLIENT, ES_VERSION

requires_es_8 = pytest.mark.skipif(
    ES_VERSION < (8, 0, 0),
    reason="This test requires at least Elasticsearch version 8.0.0",
)

requires_sklearn = pytest.mark.skipif(
    not HAS_SKLEARN, reason="This test requires 'scikit-learn' package to run"
)

requires_transforms = pytest.mark.skipif(
    not HAS_TRANSFORMERS, reason="This test requires 'transformers' package to run"
)


@pytest.fixture(scope="function", autouse=True)
def delete_test_index():
    yield
    model = PyTorchModel(ES_TEST_CLIENT, "distilbert-base-uncased")
    model.stop()
    model.delete()


class TestPytorchModel:
    @requires_es_8
    @requires_sklearn
    @requires_transforms
    def test_infer(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            print("Loading HuggingFace transformer tokenizer and model")
            tm = TransformerModel("distilbert-base-uncased", "fill_mask", False)
            model_path, config_path, vocab_path = tm.save(tmp_dir)
            ptm = PyTorchModel(ES_TEST_CLIENT, tm.elasticsearch_model_id())
            ptm.stop()
            ptm.delete()
            print(f"Importing model: {ptm.model_id}")
            ptm.import_model(model_path, config_path, vocab_path)
            ptm.start()
            result = ptm.infer(
                {"docs": [{"text_field": "[MASK] is the capital of France."}]}
            )
            assert result["predicted_value"] == "paris"
