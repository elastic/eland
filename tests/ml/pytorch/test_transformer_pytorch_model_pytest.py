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
import os
import tempfile
from abc import ABC
from typing import Union

import numpy as np
import pytest
from elasticsearch import NotFoundError

try:
    import sklearn  # noqa: F401

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import torch  # noqa: F401
    from torch import Tensor, nn  # noqa: F401

    from eland.ml.pytorch import (  # noqa: F401
        NlpBertTokenizationConfig,
        NlpTrainedModelConfig,
        PyTorchModel,
        TraceableModel,
    )
    from eland.ml.pytorch.nlp_ml_model import (
        NerInferenceOptions,
        TextClassificationInferenceOptions,
        TextEmbeddingInferenceOptions,
    )

    TracedModelTypes = Union[  # noqa: F401
        torch.nn.Module,
        torch.ScriptModule,
        torch.jit.ScriptModule,
        torch.jit.TopLevelTracedModule,
    ]

    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False

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
        not HAS_PYTORCH, reason="This test requires 'pytorch' package to run"
    ),
]

TEST_BERT_VOCAB = [
    "Elastic",
    "##search",
    "is",
    "fun",
    "my",
    "little",
    "red",
    "car",
    "God",
    "##zilla",
    ".",
    ",",
    "[CLS]",
    "[SEP]",
    "[MASK]",
    "[PAD]",
    "[UNK]",
    "day",
    "Pancake",
    "with",
    "?",
]

NER_LABELS = [
    "O",
    "B_MISC",
    "I_MISC",
    "B_PER",
    "I_PER",
    "B_ORG",
    "I_ORG",
    "B_LOC",
    "I_LOC",
]

TEXT_CLASSIFICATION_LABELS = ["foo", "bar", "baz"]

if not HAS_PYTORCH:
    pytest.skip("This test requires 'pytorch' package to run", allow_module_level=True)


class TestTraceableModel(TraceableModel, ABC):
    def __init__(self, model: nn.Module):
        super().__init__(model)

    def _trace(self) -> TracedModelTypes:
        input_ids = torch.tensor(np.array(range(0, len(TEST_BERT_VOCAB))))
        attention_mask = torch.tensor([1] * len(TEST_BERT_VOCAB))
        token_type_ids = torch.tensor([0] * len(TEST_BERT_VOCAB))
        position_ids = torch.arange(len(TEST_BERT_VOCAB), dtype=torch.long)
        return torch.jit.trace(
            self._model,
            (
                input_ids,
                attention_mask,
                token_type_ids,
                position_ids,
            ),
        )


class NerModule(nn.Module):
    def forward(
        self,
        input_ids: Tensor,
        _attention_mask: Tensor,
        _token_type_ids: Tensor,
        _position_ids: Tensor,
    ) -> Tensor:
        """Wrap the input and output to conform to the native process interface."""
        outside = [0] * len(NER_LABELS)
        outside[0] = 1
        person = [0] * len(NER_LABELS)
        person[3] = 1
        person[4] = 1
        result = [outside for _t in np.array(input_ids.data)]
        result[1] = person
        result[2] = person
        return torch.tensor([result], dtype=torch.float)


class EmbeddingModule(nn.Module):
    def forward(
        self,
        _input_ids: Tensor,
        _attention_mask: Tensor,
        _token_type_ids: Tensor,
        _position_ids: Tensor,
    ) -> Tensor:
        """Wrap the input and output to conform to the native process interface."""
        result = [0] * 512
        return torch.tensor([result], dtype=torch.float)


class TextClassificationModule(nn.Module):
    def forward(
        self,
        _input_ids: Tensor,
        _attention_mask: Tensor,
        _token_type_ids: Tensor,
        _position_ids: Tensor,
    ) -> Tensor:
        # foo, bar, baz are the classification labels
        result = [0, 1.0, 0]
        return torch.tensor([result], dtype=torch.float)


MODELS_TO_TEST = [
    (
        "ner",
        TestTraceableModel(model=NerModule()),
        NlpTrainedModelConfig(
            description="test ner model",
            inference_config=NerInferenceOptions(
                tokenization=NlpBertTokenizationConfig(),
                classification_labels=NER_LABELS,
            ),
        ),
        "Godzilla Pancake Elasticsearch is fun.",
        "[Godzilla](PER&Godzilla) Pancake Elasticsearch is fun.",
    ),
    (
        "embedding",
        TestTraceableModel(model=EmbeddingModule()),
        NlpTrainedModelConfig(
            description="test text_embedding model",
            inference_config=TextEmbeddingInferenceOptions(
                tokenization=NlpBertTokenizationConfig()
            ),
        ),
        "Godzilla Pancake Elasticsearch is fun.",
        [0] * 512,
    ),
    (
        "text_classification",
        TestTraceableModel(model=TextClassificationModule()),
        NlpTrainedModelConfig(
            description="test text_classification model",
            inference_config=TextClassificationInferenceOptions(
                tokenization=NlpBertTokenizationConfig(),
                classification_labels=TEXT_CLASSIFICATION_LABELS,
            ),
        ),
        "Godzilla Pancake Elasticsearch is fun.",
        "bar",
    ),
]


@pytest.fixture(scope="function", autouse=True)
def setup_and_tear_down():
    ES_TEST_CLIENT.cluster.put_settings(
        body={"transient": {"logger.org.elasticsearch.xpack.ml": "DEBUG"}}
    )
    yield
    for (
        model_id,
        _,
        _,
        _,
        _,
    ) in MODELS_TO_TEST:
        model = PyTorchModel(ES_TEST_CLIENT, model_id.replace("/", "__").lower()[:64])
        try:
            model.stop()
            model.delete()
        except NotFoundError:
            pass


def upload_model_and_start_deployment(
    tmp_dir: str, model: TraceableModel, config: NlpTrainedModelConfig, model_id: str
):
    print("Loading HuggingFace transformer tokenizer and model")
    model_path = model.save(tmp_dir)
    vocab_path = os.path.join(tmp_dir, "vocabulary.json")
    with open(vocab_path, "w") as outfile:
        json.dump({"vocabulary": TEST_BERT_VOCAB}, outfile)
    ptm = PyTorchModel(ES_TEST_CLIENT, model_id)
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


class TestPytorchModelUpload:
    @pytest.mark.parametrize("model_id,model,config,input,prediction", MODELS_TO_TEST)
    def test_model_upload(self, model_id, model, config, input, prediction):
        with tempfile.TemporaryDirectory() as tmp_dir:
            ptm = upload_model_and_start_deployment(tmp_dir, model, config, model_id)
            result = ptm.infer(docs=[{"text_field": input}])
            assert result.get("predicted_value") is not None
            assert result["predicted_value"] == prediction
