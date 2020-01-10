_Note, this project is still very much a work in progress and in an alpha state; input and contributions welcome!_

# eland: pandas-like Python client for analysis of Elasticsearch data

<table>
<tr>
  <td>Latest Release</td>
  <td>
    <a href="https://pypi.org/project/eland/">
    <img src="https://img.shields.io/pypi/v/eland.svg" alt="latest release" />
    </a>
  </td>
<tr>
  <td>Package Status</td>
  <td>
		<a href="https://pypi.org/project/eland/">
		<img src="https://img.shields.io/pypi/status/eland.svg" alt="status" />
		</a>
  </td>
</tr>
<tr>
  <td>License</td>
  <td>
    <a href="https://github.com/elastic/eland/LICENSE.txt">
    <img src="https://img.shields.io/pypi/l/eland.svg" alt="license" />
    </a>
</td>
</tr>
<tr>
  <td>Build Status</td>
  <td>
    <a href="https://clients-ci.elastic.co/job/elastic+eland+master/">
    <img src="https://clients-ci.elastic.co/buildStatus/icon?job=elastic%2Beland%2Bmaster" alt="Build Status" />
    </a>
  </td>
</tr>
</table>

# What is it?

eland is a Elasticsearch client Python package to analyse, explore and manipulate data that resides in Elasticsearch. 
Where possible the package uses existing Python APIs and data structures to make it easy to switch between numpy, 
pandas, scikit-learn to their Elasticsearch powered equivalents. In general, the data resides in Elasticsearch and 
not in memory, which allows eland to access large datasets stored in Elasticsearch.

For example, to explore data in a large Elasticsearch index, simply create an eland DataFrame from an Elasticsearch 
index pattern, and explore using an API that mirrors a subset of the pandas.DataFrame API: 

```
>>> import eland as ed

>>> df = ed.read_es('http://localhost:9200', 'reviews') 

>>> df.head()
   reviewerId  vendorId  rating              date
0           0         0       5  2006-04-07 17:08
1           1         1       5  2006-05-04 12:16
2           2         2       4  2006-04-21 12:26
3           3         3       5  2006-04-18 15:48
4           3         4       5  2006-04-18 15:49

>>> df.describe()
          reviewerId       vendorId         rating
count  578805.000000  578805.000000  578805.000000
mean   174124.098437      60.645267       4.679671
std    116951.972209      54.488053       0.800891
min         0.000000       0.000000       0.000000
25%     70043.000000      20.000000       5.000000
50%    161052.000000      44.000000       5.000000
75%    272697.000000      83.000000       5.000000
max    400140.000000     246.000000       5.000000
```

See [docs](https://eland.readthedocs.io/en/latest) and [demo_notebook.ipynb](https://eland.readthedocs.io/en/latest/examples/demo_notebook.html) for more examples.

## Where to get it
The source code is currently hosted on GitHub at:
https://github.com/elastic/eland

Binary installers for the latest released version are available at the [Python
package index](https://pypi.org/project/eland).

```sh
pip install eland
```

## Development Setup

1. Create a virtual environment in Python 

For example, 

```
python3 -m venv env
```

2. Activate the virtual environment

```
source env/bin/activate
```

3. Install dependencies from the `requirements.txt` file

```
pip install -r requirements.txt
```

## Versions and Compatibility

### Python Version Support

Officially Python 3.5.3 and above, 3.6, 3.7, and 3.8.

eland depends on pandas version 0.25.3. 

#### Elasticsearch Versions

eland is versioned like the Elastic stack (eland 7.5.1 is compatible with Elasticsearch 7.x up to 7.5.1)

A major version of the client is compatible with the same major version of Elasticsearch. 

No compatibility assurances are given between different major versions of the client and Elasticsearch. 
Major differences likely exist between major versions of Elasticsearch, 
particularly around request and response object formats, but also around API urls and behaviour.

## Connecting to Elasticsearch Cloud

```
>>> import eland as ed
>>> from elasticsearch import Elasticsearch

>>> es = Elasticsearch(cloud_id="<cloud_id>", http_auth=('<user>','<password>'))

>>> es.info()
{'name': 'instance-0000000000', 'cluster_name': 'bf900cfce5684a81bca0be0cce5913bc', 'cluster_uuid': 'xLPvrV3jQNeadA7oM4l1jA', 'version': {'number': '7.4.2', 'build_flavor': 'default', 'build_type': 'tar', 'build_hash': '2f90bbf7b93631e52bafb59b3b049cb44ec25e96', 'build_date': '2019-10-28T20:40:44.881551Z', 'build_snapshot': False, 'lucene_version': '8.2.0', 'minimum_wire_compatibility_version': '6.8.0', 'minimum_index_compatibility_version': '6.0.0-beta1'}, 'tagline': 'You Know, for Search'}

>>> df = ed.read_es(es, 'reviews')
```

## Why eland?

Naming is difficult, but as we had to call it something:

* eland: elastic and data
* eland: 'Elk/Moose' in Dutch (Alces alces)
* [Elandsgracht](https://goo.gl/maps/3hGBMqeGRcsBJfKx8): Amsterdam street near Elastic's Amsterdam office 

[Pronunciation](https://commons.wikimedia.org/wiki/File:Nl-eland.ogg): /ˈeːlɑnt/

