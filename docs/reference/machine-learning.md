---
mapped_pages:
  - https://www.elastic.co/guide/en/elasticsearch/client/eland/current/machine-learning.html
---

# Machine Learning [machine-learning]


## Trained models [ml-trained-models]

Eland allows transforming trained models from scikit-learn, XGBoost, and LightGBM libraries to be serialized and used as an inference model in {{es}}.

```python
>>> from xgboost import XGBClassifier
>>> from eland.ml import MLModel

# Train and exercise an XGBoost ML model locally
>>> xgb_model = XGBClassifier(booster="gbtree")
>>> xgb_model.fit(training_data[0], training_data[1])

>>> xgb_model.predict(training_data[0])
[0 1 1 0 1 0 0 0 1 0]

# Import the model into Elasticsearch
>>> es_model = MLModel.import_model(
    es_client="http://localhost:9200",
    model_id="xgb-classifier",
    model=xgb_model,
    feature_names=["f0", "f1", "f2", "f3", "f4"],
)

# Exercise the ML model in Elasticsearch with the training data
>>> es_model.predict(training_data[0])
[0 1 1 0 1 0 0 0 1 0]
```


## Natural language processing (NLP) with PyTorch [ml-nlp-pytorch]

::::{important}
You need to install the appropriate version of PyTorch to import an NLP model. Run `python -m pip install 'eland[pytorch]'` to install that version.
::::


For NLP tasks, Eland enables you to import PyTorch models into {{es}}. Use the `eland_import_hub_model` script to download and install supported [transformer models](https://huggingface.co/transformers) from the [Hugging Face model hub](https://huggingface.co/models). For example:

```bash
$ eland_import_hub_model <authentication> \ <1>
  --url http://localhost:9200/ \ <2>
  --hub-model-id elastic/distilbert-base-cased-finetuned-conll03-english \ <3>
  --task-type ner \ <4>
  --start
```

1. Use an authentication method to access your cluster. Refer to [Authentication methods](machine-learning.md#ml-nlp-pytorch-auth).
2. The cluster URL. Alternatively, use `--cloud-id`.
3. Specify the identifier for the model in the Hugging Face model hub.
4. Specify the type of NLP task. Supported values are `fill_mask`, `ner`, `question_answering`, `text_classification`, `text_embedding`, `text_expansion`, `text_similarity` and `zero_shot_classification`.


For more information about the available options, run `eland_import_hub_model` with the `--help` option.

```bash
$ eland_import_hub_model --help
```


### Import model with Docker [ml-nlp-pytorch-docker]

::::{important}
To use the Docker container, you need to clone the Eland repository: [https://github.com/elastic/eland](https://github.com/elastic/eland)
::::


If you want to use Eland without installing it, you can use the Docker image:

You can use the container interactively:

```bash
$ docker run -it --rm --network host docker.elastic.co/eland/eland
```

Running installed scripts is also possible without an interactive shell, for example:

```bash
docker run -it --rm docker.elastic.co/eland/eland \
    eland_import_hub_model \
      --url $ELASTICSEARCH_URL \
      --hub-model-id elastic/distilbert-base-uncased-finetuned-conll03-english \
      --start
```

Replace the `$ELASTICSEARCH_URL` with the URL for your Elasticsearch cluster. For authentication purposes, include an administrator username and password in the URL in the following format: `https://username:password@host:port`.


### Install models in an air-gapped environment [ml-nlp-pytorch-air-gapped]

You can install models in a restricted or closed network by pointing the `eland_import_hub_model` script to local files.

For an offline install of a Hugging Face model, the model first needs to be cloned locally, Git and [Git Large File Storage](https://git-lfs.com/) are required to be installed in your system.

1. Select a model you want to use from Hugging Face. Refer to the [compatible third party model](docs-content://explore-analyze/machine-learning/nlp/ml-nlp-model-ref.md) list for more information on the supported architectures.
2. Clone the selected model from Hugging Face by using the model URL. For example:

    ```bash
    git clone https://huggingface.co/dslim/bert-base-NER
    ```

    This command results in a local copy of of the model in the directory `bert-base-NER`.

3. Use the `eland_import_hub_model` script with the `--hub-model-id` set to the directory of the cloned model to install it:

    ```bash
    eland_import_hub_model \
          --url 'XXXX' \
          --hub-model-id /PATH/TO/MODEL \
          --task-type ner \
          --es-username elastic --es-password XXX \
          --es-model-id bert-base-ner
    ```

    If you use the Docker image to run `eland_import_hub_model` you must bind mount the model directory, so the container can read the files:

    ```bash
    docker run --mount type=bind,source=/PATH/TO/MODEL,destination=/model,readonly -it --rm docker.elastic.co/eland/eland \
        eland_import_hub_model \
          --url 'XXXX' \
          --hub-model-id /model \
          --task-type ner \
          --es-username elastic --es-password XXX \
          --es-model-id bert-base-ner
    ```

    Once it’s uploaded to {{es}}, the model will have the ID specified by `--es-model-id`. If it is not set, the model ID is derived from `--hub-model-id`; spaces and path delimiters are converted to double underscores `__`.



### Connect to Elasticsearch through a proxy [ml-nlp-pytorch-proxy]

Behind the scenes, Eland uses the `requests` Python library, which [allows configuring proxies through an environment variable](https://requests.readthedocs.io/en/latest/user/advanced/#proxies). For example, to use an HTTP proxy to connect to an HTTPS Elasticsearch cluster, you need to set the `HTTPS_PROXY` environment variable when invoking Eland:

```bash
HTTPS_PROXY=http://proxy-host:proxy-port eland_import_hub_model ...
```

If you disabled security on your Elasticsearch cluster, you should use `HTTP_PROXY` instead.


### Authentication methods [ml-nlp-pytorch-auth]

The following authentication options are available when using the import script:

* Elasticsearch username and password authentication (specified with the `-u` and `-p` options):

    ```bash
    eland_import_hub_model -u <username> -p <password> --cloud-id <cloud-id> ...
    ```

    These `-u` and `-p` options also work when you use `--url`.

* Elasticsearch username and password authentication (embedded in the URL):

    ```bash
    eland_import_hub_model --url https://<user>:<password>@<hostname>:<port> ...
    ```

* Elasticsearch API key authentication:

    ```bash
    eland_import_hub_model --es-api-key <api-key> --url https://<hostname>:<port> ...
    ```

* HuggingFace Hub access token (for private models):

    ```bash
    eland_import_hub_model --hub-access-token <access-token> ...
    ```



### TLS/SSL [ml-nlp-pytorch-tls]

The following TLS/SSL options for Elasticsearch are available when using the import script:

* Specify alternate CA bundle to verify the cluster certificate:

    ```bash
    eland_import_hub_model --ca-certs CA_CERTS ...
    ```

* Disable TLS/SSL verification altogether (strongly discouraged):

    ```bash
    eland_import_hub_model --insecure ...
    ```


