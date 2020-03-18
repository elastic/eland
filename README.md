_Note, this project is still very much a work in progress and in an alpha state; input and contributions welcome!_

<p align="center">
      <a href="https://github.com/elastic/eland">
         <img src="./docs/source/logo/eland.png" width="30%" alt="eland" />
      </a>
</p>
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
    <a href="https://github.com/elastic/eland/blob/master/LICENSE.txt">
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

>>> # Connect to 'flights' index via localhost Elasticsearch node
>>> df = ed.DataFrame('localhost:9200', 'flights') 

>>> df.head()
   AvgTicketPrice  Cancelled           Carrier  ...        OriginWeather dayOfWeek           timestamp
0      841.265642      False   Kibana Airlines  ...                Sunny         0 2018-01-01 00:00:00
1      882.982662      False  Logstash Airways  ...                Clear         0 2018-01-01 18:27:00
2      190.636904      False  Logstash Airways  ...                 Rain         0 2018-01-01 17:11:14
3      181.694216       True   Kibana Airlines  ...  Thunder & Lightning         0 2018-01-01 10:33:28
4      730.041778      False   Kibana Airlines  ...        Damaging Wind         0 2018-01-01 05:13:00

[5 rows x 27 columns]

>>> df.describe()
       AvgTicketPrice  DistanceKilometers  DistanceMiles  FlightDelayMin  FlightTimeHour  FlightTimeMin     dayOfWeek
count    13059.000000        13059.000000   13059.000000    13059.000000    13059.000000   13059.000000  13059.000000
mean       628.253689         7092.142457    4406.853010       47.335171        8.518797     511.127842      2.835975
std        266.386661         4578.263193    2844.800855       96.743006        5.579019     334.741135      1.939365
min        100.020531            0.000000       0.000000        0.000000        0.000000       0.000000      0.000000
25%        410.008918         2470.545974    1535.126118        0.000000        4.194976     251.738513      1.000000
50%        640.362667         7612.072403    4729.922470        0.000000        8.385816     503.148975      3.000000
75%        842.254990         9735.082407    6049.459005       15.000000       12.009396     720.534532      4.141221
max       1199.729004        19881.482422   12353.780273      360.000000       31.715034    1902.901978      6.000000

>>> df[['Carrier', 'AvgTicketPrice', 'Cancelled']]
                Carrier  AvgTicketPrice  Cancelled
0       Kibana Airlines      841.265642      False
1      Logstash Airways      882.982662      False
2      Logstash Airways      190.636904      False
3       Kibana Airlines      181.694216       True
4       Kibana Airlines      730.041778      False
...                 ...             ...        ...
13054  Logstash Airways     1080.446279      False
13055  Logstash Airways      646.612941      False
13056  Logstash Airways      997.751876      False
13057          JetBeats     1102.814465      False
13058          JetBeats      858.144337      False

[13059 rows x 3 columns]

>>> df[(df.Carrier=="Kibana Airlines") & (df.AvgTicketPrice > 900.0) & (df.Cancelled == True)].head()
     AvgTicketPrice  Cancelled          Carrier  ...        OriginWeather dayOfWeek           timestamp
8        960.869736       True  Kibana Airlines  ...            Heavy Fog         0 2018-01-01 12:09:35
26       975.812632       True  Kibana Airlines  ...                 Rain         0 2018-01-01 15:38:32
311      946.358410       True  Kibana Airlines  ...            Heavy Fog         0 2018-01-01 11:51:12
651      975.383864       True  Kibana Airlines  ...                 Rain         2 2018-01-03 21:13:17
950      907.836523       True  Kibana Airlines  ...  Thunder & Lightning         2 2018-01-03 05:14:51

[5 rows x 27 columns]

>>> df[['DistanceKilometers', 'AvgTicketPrice']].aggregate(['sum', 'min', 'std'])
     DistanceKilometers  AvgTicketPrice
sum        9.261629e+07    8.204365e+06
min        0.000000e+00    1.000205e+02
std        4.578263e+03    2.663867e+02

>>> df[['Carrier', 'Origin', 'Dest']].nunique()
Carrier      4
Origin     156
Dest       156
dtype: int64

>>> s = df.AvgTicketPrice * 2 + df.DistanceKilometers - df.FlightDelayMin
>>> s
0        18174.857422
1        10589.365723
2          381.273804
3          739.126221
4        14818.327637
             ...     
13054    10219.474121
13055     8381.823975
13056    12661.157104
13057    20819.488281
13058    18315.431274
Length: 13059, dtype: float64

>>> print(s.info_es())
index_pattern: flights
Index:
 index_field: _id
 is_source_field: False
Mappings:
 capabilities:
         es_field_name  is_source es_dtype es_date_format pd_dtype  is_searchable  is_aggregatable  is_scripted aggregatable_es_field_name
NaN  script_field_None      False   double           None  float64           True             True         True          script_field_None
Operations:
 tasks: []
 size: None
 sort_params: None
 _source: ['script_field_None']
 body: {'script_fields': {'script_field_None': {'script': {'source': "(((doc['AvgTicketPrice'].value * 2) + doc['DistanceKilometers'].value) - doc['FlightDelayMin'].value)"}}}}
 post_processing: []

>>> pd_df = ed.eland_to_pandas(df)
>>> pd_df.head()
   AvgTicketPrice  Cancelled           Carrier  ...        OriginWeather dayOfWeek           timestamp
0      841.265642      False   Kibana Airlines  ...                Sunny         0 2018-01-01 00:00:00
1      882.982662      False  Logstash Airways  ...                Clear         0 2018-01-01 18:27:00
2      190.636904      False  Logstash Airways  ...                 Rain         0 2018-01-01 17:11:14
3      181.694216       True   Kibana Airlines  ...  Thunder & Lightning         0 2018-01-01 10:33:28
4      730.041778      False   Kibana Airlines  ...        Damaging Wind         0 2018-01-01 05:13:00

[5 rows x 27 columns]
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

## Versions and Compatibility

### Python Version Support

Officially Python 3.5.3 and above, 3.6, 3.7, and 3.8.

eland depends on pandas version 0.25.3. 

### Elasticsearch Versions

eland is versioned like the Elastic stack (eland 7.5.1 is compatible with Elasticsearch 7.x up to 7.5.1)

A major version of the client is compatible with the same major version of Elasticsearch. 

No compatibility assurances are given between different major versions of the client and Elasticsearch. 
Major differences likely exist between major versions of Elasticsearch, 
particularly around request and response object formats, but also around API urls and behaviour.

## Connecting to Elasticsearch 

eland uses the [Elasticsearch low level client](https://elasticsearch-py.readthedocs.io/) to connect to Elasticsearch. 
This client supports a range of [connection options and authentication mechanisms]
(https://elasticsearch-py.readthedocs.io/en/master/api.html#elasticsearch). 

### Basic Connection Options

```
>>> import eland as ed

>>> # Connect to flights index via localhost Elasticsearch node
>>> ed.DataFrame('localhost', 'flights')

>>> # Connect to flights index via localhost Elasticsearch node on port 9200
>>> ed.DataFrame('localhost:9200', 'flights')

>>> # Connect to flights index via localhost Elasticsearch node on port 9200 with <user>:<password> credentials
>>> ed.DataFrame('http://<user>:<password>@localhost:9200', 'flights')

>>> # Connect to flights index via ssl
>>> es = Elasticsearch(
    'https://<user>:<password>@localhost:443', 
    use_ssl=True, 
    verify_certs=True, 
    ca_certs='/path/to/ca.crt'
)
>>> ed.DataFrame(es, 'flights')

>>> # Connect to flights index via ssl using Urllib3HttpConnection options 
>>> es = Elasticsearch(
    ['localhost:443', 'other_host:443'],
    use_ssl=True,
    verify_certs=True,
    ca_certs='/path/to/CA_certs',
    client_cert='/path/to/clientcert.pem',
    client_key='/path/to/clientkey.pem'
)
>>> ed.DataFrame(es, 'flights')
```

### Connecting to an Elasticsearch Cloud Cluster

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

