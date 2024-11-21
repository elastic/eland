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
This will create local cached copies that will be traced (necessary) before
uploading to Elasticsearch. This will also check that the task type is supported
as well as the model and tokenizer types. All necessary configuration is
uploaded along with the model.
"""
import argparse
import logging
import os
import sys
import tempfile
import textwrap

from elastic_transport.client_utils import DEFAULT
from elasticsearch import AuthenticationException, Elasticsearch

from eland._version import __version__
from eland.common import is_serverless_es, parse_es_version

MODEL_HUB_URL = "https://huggingface.co"


def get_arg_parser():
    from eland.ml.pytorch.transformers import SUPPORTED_TASK_TYPES

    parser = argparse.ArgumentParser(
        exit_on_error=False
    )  # throw exception rather than exit
    location_args = parser.add_mutually_exclusive_group(required=True)
    location_args.add_argument(
        "--url",
        default=os.environ.get("ES_URL"),
        help="An Elasticsearch connection URL, e.g. http://localhost:9200",
    )
    location_args.add_argument(
        "--cloud-id",
        default=os.environ.get("CLOUD_ID"),
        help="Cloud ID as found in the 'Manage Deployment' page of an Elastic Cloud deployment",
    )
    parser.add_argument(
        "--hub-model-id",
        required=True,
        help="The model ID in the Hugging Face model hub, "
        "e.g. dbmdz/bert-large-cased-finetuned-conll03-english",
    )
    parser.add_argument(
        "--hub-access-token",
        required=False,
        default=os.environ.get("HUB_ACCESS_TOKEN"),
        help="The Hugging Face access token, needed to access private models",
    )
    parser.add_argument(
        "--es-model-id",
        required=False,
        default=None,
        help="The model ID to use in Elasticsearch, "
        "e.g. bert-large-cased-finetuned-conll03-english."
        "When left unspecified, this will be auto-created from the `hub-id`",
    )
    parser.add_argument(
        "-u",
        "--es-username",
        required=False,
        default=os.environ.get("ES_USERNAME"),
        help="Username for Elasticsearch",
    )
    parser.add_argument(
        "-p",
        "--es-password",
        required=False,
        default=os.environ.get("ES_PASSWORD"),
        help="Password for the Elasticsearch user specified with -u/--username",
    )
    parser.add_argument(
        "--es-api-key",
        required=False,
        default=os.environ.get("ES_API_KEY"),
        help="API key for Elasticsearch",
    )
    parser.add_argument(
        "--task-type",
        required=False,
        choices=SUPPORTED_TASK_TYPES,
        help="The task type for the model usage. Use text_similarity for rerank tasks. Will attempt to auto-detect task type for the model if not provided. "
        "Default: auto",
        default="auto",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        default=False,
        help="Quantize the model before uploading. Default: False",
    )
    parser.add_argument(
        "--start",
        action="store_true",
        default=False,
        help="Start the model deployment after uploading. Default: False",
    )
    parser.add_argument(
        "--clear-previous",
        action="store_true",
        default=False,
        help="Should the model previously stored with `es-model-id` be deleted",
    )
    parser.add_argument(
        "--insecure",
        action="store_false",
        default=True,
        help="Do not verify SSL certificates",
    )
    parser.add_argument(
        "--ca-certs", required=False, default=DEFAULT, help="Path to CA bundle"
    )

    parser.add_argument(
        "--ingest-prefix",
        required=False,
        default=None,
        help="String to prepend to model input at ingest",
    )
    parser.add_argument(
        "--search-prefix",
        required=False,
        default=None,
        help="String to prepend to model input at search",
    )

    parser.add_argument(
        "--max-model-input-length",
        required=False,
        default=None,
        help="""Set the model's max input length.
                Usually the max input length is derived from the Hugging Face
                model confifguation. Use this option to explicity set the model's
                max input length if the value can not be found in the Hugging
                Face configuration. Max input length should never exceed the
                model's true max length, setting a smaller max length is valid.
                """,
        type=int,
    )

    return parser


def parse_args():
    parser = get_arg_parser()
    try:
        return parser.parse_args()
    except argparse.ArgumentError as argument_error:
        if argument_error.argument_name == "--task-type":
            message = (
                argument_error.message
                + "\n\nUse 'text_similarity' for rerank tasks in Elasticsearch"
            )
            parser.error(message=message)
        else:
            parser.error(message=argument_error.message)
    except argparse.ArgumentTypeError as type_error:
        parser.error(str(type_error))


def get_es_client(cli_args, logger):
    try:
        es_args = {
            "request_timeout": 300,
            "verify_certs": cli_args.insecure,
            "ca_certs": cli_args.ca_certs,
            "node_class": "requests",
        }

        # Deployment location
        if cli_args.url:
            es_args["hosts"] = cli_args.url

        if cli_args.cloud_id:
            es_args["cloud_id"] = cli_args.cloud_id

        # Authentication
        if cli_args.es_api_key:
            es_args["api_key"] = cli_args.es_api_key
        elif cli_args.es_username:
            if not cli_args.es_password:
                logging.error(
                    f"Password for user {cli_args.es_username} was not specified."
                )
                exit(1)

            es_args["basic_auth"] = (cli_args.es_username, cli_args.es_password)

        es_client = Elasticsearch(**es_args)
        return es_client
    except AuthenticationException as e:
        logger.error(e)
        exit(1)


def check_cluster_version(es_client, logger):
    es_info = es_client.info()

    if is_serverless_es(es_client):
        logger.info(f"Connected to serverless cluster '{es_info['cluster_name']}'")
        # Serverless is compatible
        # Return the latest known semantic version, i.e. this version
        return parse_es_version(__version__)

    # check the semantic version for none serverless clusters
    logger.info(
        f"Connected to cluster named '{es_info['cluster_name']}' (version: {es_info['version']['number']})"
    )

    sem_ver = parse_es_version(es_info["version"]["number"])
    major_version = sem_ver[0]

    # NLP models added in 8
    if major_version < 8:
        logger.error(
            f"Elasticsearch version {major_version} does not support NLP models. Please upgrade Elasticsearch to the latest version"
        )
        exit(1)

    # PyTorch was upgraded to version 2.3.1 in 8.15.2
    # and is incompatible with earlier versions
    if sem_ver < (8, 15, 2):
        import torch

        logger.error(
            f"Eland uses PyTorch version {torch.__version__} which is incompatible with Elasticsearch versions prior to 8.15.2. Please upgrade Elasticsearch to at least version 8.15.2"
        )
        exit(1)

    return sem_ver


def main():
    # Configure logging
    logging.basicConfig(format="%(asctime)s %(levelname)s : %(message)s")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    try:
        from eland.ml.pytorch import PyTorchModel
        from eland.ml.pytorch.transformers import (
            SUPPORTED_TASK_TYPES,
            TaskTypeError,
            TransformerModel,
            UnknownModelInputSizeError,
        )
    except ModuleNotFoundError as e:
        logger.error(
            textwrap.dedent(
                f"""\
            \033[31mFailed to run because module '{e.name}' is not available.\033[0m

            This script requires PyTorch extras to run. You can install these by running:

                \033[1m{sys.executable} -m pip install 'eland[pytorch]'
            \033[0m"""
            )
        )
        exit(1)
    assert SUPPORTED_TASK_TYPES

    # Parse arguments
    args = parse_args()

    # Connect to ES
    logger.info("Establishing connection to Elasticsearch")
    es = get_es_client(args, logger)
    cluster_version = check_cluster_version(es, logger)

    # Trace and save model, then upload it from temp file
    with tempfile.TemporaryDirectory() as tmp_dir:
        logger.info(
            f"Loading HuggingFace transformer tokenizer and model '{args.hub_model_id}'"
        )

        try:
            tm = TransformerModel(
                model_id=args.hub_model_id,
                access_token=args.hub_access_token,
                task_type=args.task_type,
                es_version=cluster_version,
                quantize=args.quantize,
                ingest_prefix=args.ingest_prefix,
                search_prefix=args.search_prefix,
                max_model_input_size=args.max_model_input_length,
            )
            model_path, config, vocab_path = tm.save(tmp_dir)
        except TaskTypeError as err:
            logger.error(
                f"Failed to get model for task type, please provide valid task type via '--task-type' parameter. Caused by {err}"
            )
            exit(1)
        except UnknownModelInputSizeError as err:
            logger.error(
                f"""Could not automatically determine the model's max input size from the model configuration.
                Please provde the max input size via the --max-model-input-length parameter. Caused by {err}"""
            )
            exit(1)

        ptm = PyTorchModel(
            es, args.es_model_id if args.es_model_id else tm.elasticsearch_model_id()
        )
        model_exists = (
            es.options(ignore_status=404)
            .ml.get_trained_models(model_id=ptm.model_id)
            .meta.status
            == 200
        )

        if model_exists:
            if args.clear_previous:
                logger.info(f"Stopping deployment for model with id '{ptm.model_id}'")
                ptm.stop()

                logger.info(f"Deleting model with id '{ptm.model_id}'")
                ptm.delete()
            else:
                logger.error(f"Trained model with id '{ptm.model_id}' already exists")
                logger.info(
                    "Run the script with the '--clear-previous' flag if you want to overwrite the existing model."
                )
                exit(1)

        logger.info(f"Creating model with id '{ptm.model_id}'")
        ptm.put_config(config=config)

        logger.info("Uploading model definition")
        ptm.put_model(model_path)

        logger.info("Uploading model vocabulary")
        ptm.put_vocab(vocab_path)

    # Start the deployed model
    if args.start:
        logger.info("Starting model deployment")
        ptm.start()

    logger.info(f"Model successfully imported with id '{ptm.model_id}'")


if __name__ == "__main__":
    main()
