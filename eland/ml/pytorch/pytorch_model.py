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
from eland.common import ensure_es_client
from tqdm import tqdm
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union

if TYPE_CHECKING:
    from elasticsearch import Elasticsearch

DEFAULT_CHUNK_SIZE = 4 * 1024 * 1024  # 4MB
QUEUE_SIZE = 4
THREAD_COUNT = 4


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
        # Elasticsearch model IDs need to be a specific format: no special chars, all lowercase, max 64 chars
        self.model_id = model_id.replace('/', '__').lower()[:64]

    @staticmethod
    def _load_json(path: str) -> Dict[str, Any]:
        with open(path, 'r') as infile:
            return json.load(infile)

    def _upload_config(self, path: str) -> bool:
        config = PyTorchModel._load_json(path)
        response = self._client.ml.put_trained_model(model_id=self.model_id, body=config)
        # TODO: Check actual result code in response
        return response is not None

    def _upload_vocab(self, path: str) -> bool:
        vocab = PyTorchModel._load_json(path)
        return self._client.transport.perform_request(
            method='PUT', url=f'/_ml/trained_models/{self.model_id}/vocabulary', body=vocab)

    def _upload_model(self, model_path: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> bool:
        file_stats = os.stat(model_path)
        total_parts = math.ceil(file_stats.st_size / chunk_size)

        def model_file_chunk_generator():
            with open(model_path, 'rb') as f:
                while True:
                    data = f.read(chunk_size)
                    if not data:
                        break
                    yield base64.b64encode(data).decode()

        for i, data in tqdm(enumerate(model_file_chunk_generator(), start=0), total=total_parts):
            body = {
                'total_definition_length': file_stats.st_size,
                'total_parts': total_parts,
                'definition': data,
            }
            self._client.transport.perform_request(
                method='PUT', url=f'/_ml/trained_models/{self.model_id}/definition/{i}', body=body)

        return True

    def upload(self, model_path: str, config_path: str, vocab_path: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> bool:
        # TODO: Implement some pre-flight checks on config, vocab, and model
        return self._upload_config(config_path) and \
               self._upload_vocab(vocab_path) and \
               self._upload_model(model_path, chunk_size)

    def start(self) -> bool:
        return self._client.transport.perform_request(
            method='POST',
            url=f'/_ml/trained_models/{self.model_id}/deployment/_start',
            params={'timeout': '60s', 'wait_for': 'started'}
        )

    def stop(self, ignore_not_found=False) -> bool:
        if ignore_not_found:
            ignorables = 404
        else:
            ignorables = ()

        return self._client.transport.perform_request(
            method='POST',
            url=f'/_ml/trained_models/{self.model_id}/deployment/_stop',
            params={'ignore': ignorables},
        )

    def delete(self, ignore_not_found=False) -> Dict[str, Any]:
        if ignore_not_found:
            ignorables = 404
        else:
            ignorables = ()

        return self._client.ml.delete_trained_model(self.model_id, ignore=ignorables)
