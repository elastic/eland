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
Helps manage PyTorch models from the command line.
"""

import argparse

import elasticsearch
import urllib3

from eland.ml.pytorch import PyTorchModel

DEFAULT_URL = "http://elastic:changeme@localhost:9200"

# For secure, self-signed localhost, disable warnings
urllib3.disable_warnings()


def _list_cmd(args: argparse.Namespace, es: elasticsearch.Elasticsearch):
    model_ids = PyTorchModel.list(es)
    print(f"Models ({len(model_ids)})")
    for model_id in model_ids:
        print(f" - {model_id}")


def _remove_all_cmd(args: argparse.Namespace, es: elasticsearch.Elasticsearch):
    model_ids = PyTorchModel.list(es)
    print(f"Removing models ({len(model_ids)})")
    for model_id in model_ids:
        print(f" - {model_id}")
        model = PyTorchModel(es, model_id)
        model.stop()
        model.delete()


def _remove_cmd(args: argparse.Namespace, es: elasticsearch.Elasticsearch):
    print(f"Removing model: {args.model}")
    model = PyTorchModel(es, args.model)
    model.stop()
    model.delete()


def main():
    parser = argparse.ArgumentParser(prog="pytorch_models.py")
    subparsers = parser.add_subparsers()
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="An Elasticsearch connection URL, e.g. http://user:secret@localhost:9200",
    )
    list_parser = subparsers.add_parser("list", description="Lists all PyTorch models")
    list_parser.set_defaults(func=_list_cmd)
    remove_all_parser = subparsers.add_parser(
        "remove-all", description="Removes all PyTorch models"
    )
    remove_all_parser.set_defaults(func=_remove_all_cmd)
    remove_parser = subparsers.add_parser(
        "remove", description="Removes a PyTorch model by ID"
    )
    remove_parser.add_argument("model", help="The model to remove")
    remove_parser.set_defaults(func=_remove_cmd)
    args = parser.parse_args()

    es = elasticsearch.Elasticsearch(
        args.url, verify_certs=False, timeout=300
    )  # 5 minute timeout

    args.func(args, es)


if __name__ == "__main__":
    main()
