---
mapped_pages:
  - https://www.elastic.co/guide/en/elasticsearch/client/eland/current/index.html
  - https://www.elastic.co/guide/en/elasticsearch/client/eland/current/overview.html
navigation_title: Eland
---

# Eland Python client [overview]

Eland is a Python client and toolkit for DataFrames and {{ml}} in {{es}}. Full documentation is available on [Read the Docs](https://eland.readthedocs.io). Source code is available on [GitHub](https://github.com/elastic/eland).


## Compatibility [_compatibility]

* Supports Python 3.9+ and Pandas 1.5
* Supports {{es}} 8+ clusters, recommended 8.16 or later for all features to work. Make sure your Eland major version matches the major version of your Elasticsearch cluster.

The recommended way to set your requirements in your `setup.py` or `requirements.txt` is::

```
# Elasticsearch 8.x
eland>=8,<9
```
```
# Elasticsearch 7.x
eland>=7,<8
```

## Getting Started [_getting_started]

Create a `DataFrame` object connected to an {{es}} cluster running on `http://localhost:9200`:

```python
>>> import eland as ed
>>> df = ed.DataFrame(
...    es_client="http://localhost:9200",
...    es_index_pattern="flights",
... )
>>> df
       AvgTicketPrice  Cancelled  ... dayOfWeek           timestamp
0          841.265642      False  ...         0 2018-01-01 00:00:00
1          882.982662      False  ...         0 2018-01-01 18:27:00
2          190.636904      False  ...         0 2018-01-01 17:11:14
3          181.694216       True  ...         0 2018-01-01 10:33:28
4          730.041778      False  ...         0 2018-01-01 05:13:00
...               ...        ...  ...       ...                 ...
13054     1080.446279      False  ...         6 2018-02-11 20:42:25
13055      646.612941      False  ...         6 2018-02-11 01:41:57
13056      997.751876      False  ...         6 2018-02-11 04:09:27
13057     1102.814465      False  ...         6 2018-02-11 08:28:21
13058      858.144337      False  ...         6 2018-02-11 14:54:34

[13059 rows x 27 columns]
```


### Elastic Cloud [_elastic_cloud]

You can also connect Eland to an Elasticsearch instance in Elastic Cloud:

```python
>>> import eland as ed
>>> from elasticsearch import Elasticsearch

# First instantiate an 'Elasticsearch' instance connected to Elastic Cloud
>>> es = Elasticsearch(cloud_id="...", api_key="...")

# then wrap the client in an Eland DataFrame:
>>> df = ed.DataFrame(es, es_index_pattern="flights")
>>> df.head(5)
       AvgTicketPrice  Cancelled  ... dayOfWeek           timestamp
0          841.265642      False  ...         0 2018-01-01 00:00:00
1          882.982662      False  ...         0 2018-01-01 18:27:00
2          190.636904      False  ...         0 2018-01-01 17:11:14
3          181.694216       True  ...         0 2018-01-01 10:33:28
4          730.041778      False  ...         0 2018-01-01 05:13:00
[5 rows x 27 columns]
```

Eland can be used for complex queries and aggregations:

```python
>>> df[df.Carrier != "Kibana Airlines"].groupby("Carrier").mean(numeric_only=False)
                  AvgTicketPrice  Cancelled                     timestamp
Carrier
ES-Air                630.235816   0.129814 2018-01-21 20:45:00.200000000
JetBeats              627.457373   0.134698 2018-01-21 14:43:18.112400635
Logstash Airways      624.581974   0.125188 2018-01-21 16:14:50.711798340
```

