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
Copies a model from the Hugging Face model hub into an Elasticsearch cluster.
This will create local cached copies that will be traced (necessary) and
optionally quantized before uploading to Elasticsearch. This will also check
the task type is supported as well as the model and tokenizer types. All
necessary configuration is uploaded along with the model.
"""

import argparse
import elasticsearch
import os
import sys
import tempfile

# project library
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from eland.ml.pytorch.external_models import *
from eland.ml.pytorch.pytorch_model import *

DEFAULT_URL = 'http://elastic:changeme@localhost:9200'
MODEL_HUB_URL = 'https://huggingface.co'


def main():
    parser = argparse.ArgumentParser(prog='upload_hub_model.py')
    parser.add_argument('--url', default=DEFAULT_URL,
                        help="An Elasticsearch connection URL, e.g. http://user:secret@localhost:9200")
    parser.add_argument('--model-id', required=True,
                        help="The model ID in the Hugging Face model hub, "
                             "e.g. dbmdz/bert-large-cased-finetuned-conll03-english")
    parser.add_argument('--task-type', required=True, choices=SUPPORTED_TASK_TYPES,
                        help=f"The task type that the model will be used for.")
    parser.add_argument('--start', action='store_true', default=False,
                        help="Start the model deployment after uploading. Default: False")
    args = parser.parse_args()

    es = elasticsearch.Elasticsearch(args.url, timeout=300)  # 5 minute timeout

    # trace and save model, then upload it from temp file
    with tempfile.TemporaryDirectory() as tmp_dir:
        print("Loading HuggingFace transformer tokenizer and model")
        hftm = HuggingFaceTransformerModel(args.model_id, args.task_type)
        model_path, config_path, vocab_path = hftm.save(tmp_dir)

        ptm = PyTorchModel(es, hftm.es_model_id)
        ptm.stop()
        ptm.delete()
        print("Uploading model")
        ptm.upload(model_path, config_path, vocab_path)

    # start the deployed model
    if args.start:
        print("Starting model deployment")
        if ptm.start():
            print(f" - started: {ptm.model_id}")
        else:
            print(f" - failed")


if __name__ == '__main__':
    main()
