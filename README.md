<div align="center">
  <a href="https://github.com/elastic/eland">
    <img src="https://raw.githubusercontent.com/elastic/eland/master/docs/sphinx/logo/eland.png" width="30%"
      alt="Eland" />
  </a>
</div>
<br />
<div align="center">
  <a href="https://pypi.org/project/eland"><img src="https://img.shields.io/pypi/v/eland.svg" alt="PyPI Version"></a>
  <a href="https://anaconda.org/conda-forge/eland"><img src="https://img.shields.io/conda/vn/conda-forge/eland"
      alt="Conda Version"></a>
  <a href="https://pepy.tech/project/eland"><img src="https://pepy.tech/badge/eland" alt="Downloads"></a>
  <a href="https://pypi.org/project/eland"><img src="https://img.shields.io/pypi/status/eland.svg"
      alt="Package Status"></a>
  <a href="https://clients-ci.elastic.co/job/elastic+eland+master"><img
      src="https://clients-ci.elastic.co/buildStatus/icon?job=elastic%2Beland%2Bmaster" alt="Build Status"></a>
  <a href="https://github.com/elastic/eland/blob/master/LICENSE.txt"><img src="https://img.shields.io/pypi/l/eland.svg"
      alt="License"></a>
  <a href="https://eland.readthedocs.io"><img
      src="https://readthedocs.org/projects/eland/badge/?version=latest" alt="Documentation Status"></a>
</div>

## About

Eland is a Python Elasticsearch client for exploring and
analyzing data in Elasticsearch with a familiar Pandas-compatible API.

Where possible the package uses existing Python APIs and data structures to make it easy to switch between numpy,
pandas, scikit-learn to their Elasticsearch powered equivalents. In general, the data resides in Elasticsearch and
not in memory, which allows Eland to access large datasets stored in Elasticsearch.

Eland also provides tools to upload trained machine learning models from your
common libraries like [scikit-learn](https://scikit-learn.org), [XGBoost](https://xgboost.readthedocs.io),
and [LightGBM](https://lightgbm.readthedocs.io) into Elasticsearch.

## Getting Started

Eland can be installed from [PyPI](https://pypi.org/project/eland) with Pip:

```bash
$ python -m pip install eland
```

Eland can also be installed from [Conda Forge](https://anaconda.org/conda-forge/eland) with Conda:

```bash
$ conda install -c conda-forge eland
```

### Compatibility

- Supports Python 3.7+ and Pandas 1.3
- Supports Elasticsearch clusters that are 7.11+, recommended 7.14 or later for all features to work.

### Connecting to Elasticsearch 

Eland uses the [Elasticsearch low level client](https://elasticsearch-py.readthedocs.io) to connect to Elasticsearch. 
This client supports a range of [connection options and authentication options](https://elasticsearch-py.readthedocs.io/en/master/api.html#elasticsearch). 

You can pass either an instance of `elasticsearch.Elasticsearch` to Eland APIs
or a string containing the host to connect to:

```python
import eland as ed

# Connecting to an Elasticsearch instance running on 'localhost:9200'
df = ed.DataFrame("localhost:9200", es_index_pattern="flights")

# Connecting to an Elastic Cloud instance
from elasticsearch import Elasticsearch

es = Elasticsearch(
    cloud_id="cluster-name:...",
    http_auth=("elastic", "<password>")
)
df = ed.DataFrame(es, es_index_pattern="flights")
```

## DataFrames in Eland

`eland.DataFrame` wraps an Elasticsearch index in a Pandas-like API
and defers all processing and filtering of data to Elasticsearch
instead of your local machine. This means you can process large
amounts of data within Elasticsearch from a Jupyter Notebook
without overloading your machine.

➤ [Eland DataFrame API documentation](https://eland.readthedocs.io/en/latest/reference/dataframe.html)

➤ [Advanced examples in a Jupyter Notebook](https://eland.readthedocs.io/en/latest/examples/demo_notebook.html)

```python
>>> import eland as ed

>>> # Connect to 'flights' index via localhost Elasticsearch node
>>> df = ed.DataFrame('localhost:9200', 'flights')

# eland.DataFrame instance has the same API as pandas.DataFrame
# except all data is in Elasticsearch. See .info() memory usage.
>>> df.head()
   AvgTicketPrice  Cancelled  ... dayOfWeek           timestamp
0      841.265642      False  ...         0 2018-01-01 00:00:00
1      882.982662      False  ...         0 2018-01-01 18:27:00
2      190.636904      False  ...         0 2018-01-01 17:11:14
3      181.694216       True  ...         0 2018-01-01 10:33:28
4      730.041778      False  ...         0 2018-01-01 05:13:00

[5 rows x 27 columns]

>>> df.info()
<class 'eland.dataframe.DataFrame'>
Index: 13059 entries, 0 to 13058
Data columns (total 27 columns):
 #   Column              Non-Null Count  Dtype         
---  ------              --------------  -----         
 0   AvgTicketPrice      13059 non-null  float64       
 1   Cancelled           13059 non-null  bool          
 2   Carrier             13059 non-null  object        
...      
 24  OriginWeather       13059 non-null  object        
 25  dayOfWeek           13059 non-null  int64         
 26  timestamp           13059 non-null  datetime64[ns]
dtypes: bool(2), datetime64[ns](1), float64(5), int64(2), object(17)
memory usage: 80.0 bytes
Elasticsearch storage usage: 5.043 MB

# Filtering of rows using comparisons
>>> df[(df.Carrier=="Kibana Airlines") & (df.AvgTicketPrice > 900.0) & (df.Cancelled == True)].head()
     AvgTicketPrice  Cancelled  ... dayOfWeek           timestamp
8        960.869736       True  ...         0 2018-01-01 12:09:35
26       975.812632       True  ...         0 2018-01-01 15:38:32
311      946.358410       True  ...         0 2018-01-01 11:51:12
651      975.383864       True  ...         2 2018-01-03 21:13:17
950      907.836523       True  ...         2 2018-01-03 05:14:51

[5 rows x 27 columns]

# Running aggregations across an index
>>> df[['DistanceKilometers', 'AvgTicketPrice']].aggregate(['sum', 'min', 'std'])
     DistanceKilometers  AvgTicketPrice
sum        9.261629e+07    8.204365e+06
min        0.000000e+00    1.000205e+02
std        4.578263e+03    2.663867e+02
```

## Machine Learning in Eland

Eland allows transforming trained models from scikit-learn, XGBoost, and LightGBM libraries
to be serialized and used as an inference model in Elasticsearch

➤ [Eland Machine Learning API documentation](https://eland.readthedocs.io/en/latest/reference/ml.html)

➤ [Read more about Machine Learning in Elasticsearch](https://www.elastic.co/guide/en/machine-learning/current/ml-getting-started.html)

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
    es_client="localhost:9200",
    model_id="xgb-classifier",
    model=xgb_model,
    feature_names=["f0", "f1", "f2", "f3", "f4"],
)

# Exercise the ML model in Elasticsearch with the training data
>>> es_model.predict(training_data[0])
[0 1 1 0 1 0 0 0 1 0]
```
