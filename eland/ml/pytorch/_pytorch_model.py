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

import base64
import json
import math
import os
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Set, Tuple, Union

from tqdm.auto import tqdm

from eland.common import ensure_es_client

if TYPE_CHECKING:
    from elasticsearch import Elasticsearch

DEFAULT_CHUNK_SIZE = 4 * 1024 * 1024  # 4MB
DEFAULT_TIMEOUT = "60s"


class PyTorchModel:
    """
    A PyTorch model managed by Elasticsearch.

    These models must be trained outside of Elasticsearch, conform to the
    support tokenization and inference interfaces, and exported as their
    TorchScript representations.
    """

    def __init__(
        self,
        es_client: Union[str, List[str], Tuple[str, ...], "Elasticsearch"],
        model_id: str,
    ):
        self._client = ensure_es_client(es_client)
        self.model_id = model_id

    def put_config(self, path: str) -> None:
        with open(path) as f:
            config = json.load(f)
        self._client.ml.put_trained_model(model_id=self.model_id, body=config)

    def put_vocab(self, path: str) -> None:
        with open(path) as f:
            vocab = json.load(f)
        self._client.transport.perform_request(
            "PUT",
            f"/_ml/trained_models/{self.model_id}/vocabulary",
            body=vocab,
        )

    def put_model(self, model_path: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> None:
        model_size = os.stat(model_path).st_size
        total_parts = math.ceil(model_size / chunk_size)

        def model_file_chunk_generator() -> Iterable[str]:
            with open(model_path, "rb") as f:
                while True:
                    data = f.read(chunk_size)
                    if not data:
                        break
                    yield base64.b64encode(data).decode()

        for i, data in tqdm(enumerate(model_file_chunk_generator()), total=total_parts):
            body = {
                "total_definition_length": model_size,
                "total_parts": total_parts,
                "definition": data,
            }
            self._client.transport.perform_request(
                "PUT",
                f"/_ml/trained_models/{self.model_id}/definition/{i}",
                body=body,
            )

    def import_model(
        self,
        model_path: str,
        config_path: str,
        vocab_path: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> None:
        # TODO: Implement some pre-flight checks on config, vocab, and model
        self.put_config(config_path)
        self.put_vocab(vocab_path)
        self.put_model(model_path, chunk_size)

    def infer(
        self, body: Dict[str, Any], timeout: str = DEFAULT_TIMEOUT
    ) -> Dict[str, Any]:
        return self._client.transport.perform_request(
            "POST",
            f"/_ml/trained_models/{self.model_id}/deployment/_infer",
            body=body,
            params={"timeout": timeout},
        )

    def start(self, timeout: str = DEFAULT_TIMEOUT) -> None:
        self._client.transport.perform_request(
            "POST",
            f"/_ml/trained_models/{self.model_id}/deployment/_start",
            params={"timeout": timeout, "wait_for": "started"},
        )

    def stop(self) -> None:
        self._client.transport.perform_request(
            "POST",
            f"/_ml/trained_models/{self.model_id}/deployment/_stop",
            params={"ignore": 404},
        )

    def delete(self) -> None:
        self._client.ml.delete_trained_model(self.model_id, ignore=(404,))

    @classmethod
    def list(
        cls, es_client: Union[str, List[str], Tuple[str, ...], "Elasticsearch"]
    ) -> Set[str]:
        client = ensure_es_client(es_client)
        res = client.ml.get_trained_models(model_id="*", allow_no_match=True)
        return set(
            [
                model["model_id"]
                for model in res["trained_model_configs"]
                if model["model_type"] == "pytorch"
            ]
        )
