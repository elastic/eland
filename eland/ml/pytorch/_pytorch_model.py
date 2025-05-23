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
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
)

from tqdm.auto import tqdm  # type: ignore

from eland.common import ensure_es_client
from eland.ml.pytorch.nlp_ml_model import NlpTrainedModelConfig

if TYPE_CHECKING:
    from elasticsearch import Elasticsearch

from elasticsearch._sync.client.utils import _quote

DEFAULT_CHUNK_SIZE = 1024 * 1024  # 1MB
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
        self._client: Elasticsearch = ensure_es_client(es_client)
        self.model_id = model_id

    def put_config(
        self, path: Optional[str] = None, config: Optional[NlpTrainedModelConfig] = None
    ) -> None:
        if path is not None and config is not None:
            raise ValueError("Only include path or config. Not both")
        if path is not None:
            with open(path) as f:
                config_map = json.load(f)
        elif config is not None:
            config_map = config.to_dict()
        else:
            raise ValueError("Must provide path or config")
        self._client.ml.put_trained_model(model_id=self.model_id, **config_map)

    def put_vocab(self, path: str) -> None:
        with open(path) as f:
            vocab = json.load(f)
        self._client.perform_request(
            method="PUT",
            path=f"/_ml/trained_models/{self.model_id}/vocabulary",
            headers={"accept": "application/json", "content-type": "application/json"},
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

        for i, data in tqdm(
            enumerate(model_file_chunk_generator()), unit=" parts", total=total_parts
        ):
            self._client.ml.put_trained_model_definition_part(
                model_id=self.model_id,
                part=i,
                total_definition_length=model_size,
                total_parts=total_parts,
                definition=data,
            )

    def import_model(
        self,
        *,
        model_path: str,
        config_path: Optional[str],
        vocab_path: str,
        config: Optional[NlpTrainedModelConfig] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> None:
        self.put_config(path=config_path, config=config)
        self.put_model(model_path, chunk_size)
        self.put_vocab(vocab_path)

    def infer(
        self,
        docs: List[Mapping[str, str]],
        inference_config: Optional[Mapping[str, Any]] = None,
        timeout: str = DEFAULT_TIMEOUT,
    ) -> Any:
        if docs is None:
            raise ValueError("Empty value passed for parameter 'docs'")

        __body: Dict[str, Any] = {}
        __body["docs"] = docs
        if inference_config is not None:
            __body["inference_config"] = inference_config

        __path = f"/_ml/trained_models/{_quote(self.model_id)}/_infer"
        __query: Dict[str, Any] = {}
        __query["timeout"] = timeout
        __headers = {"accept": "application/json", "content-type": "application/json"}

        return self._client.options(request_timeout=60).perform_request(
            "POST", __path, params=__query, headers=__headers, body=__body
        )

    def start(self, timeout: str = DEFAULT_TIMEOUT) -> None:
        self._client.options(request_timeout=60).ml.start_trained_model_deployment(
            model_id=self.model_id, timeout=timeout, wait_for="started"
        )

    def stop(self) -> None:
        self._client.ml.stop_trained_model_deployment(model_id=self.model_id)

    def delete(self) -> None:
        self._client.options(ignore_status=404).ml.delete_trained_model(
            model_id=self.model_id
        )

    @classmethod
    def list(
        cls, es_client: Union[str, List[str], Tuple[str, ...], "Elasticsearch"]
    ) -> Set[str]:
        client = ensure_es_client(es_client)
        resp = client.ml.get_trained_models(model_id="*", allow_no_match=True)
        return set(
            [
                model["model_id"]
                for model in resp["trained_model_configs"]
                if model["model_type"] == "pytorch"
            ]
        )
