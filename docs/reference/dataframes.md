---
mapped_pages:
  - https://www.elastic.co/guide/en/elasticsearch/client/eland/current/dataframes.html
---

# Data Frames [dataframes]

`eland.DataFrame` wraps an Elasticsearch index in a Pandas-like API and defers all processing and filtering of data to Elasticsearch instead of your local machine. This means you can process large amounts of data within Elasticsearch from a Jupyter Notebook without overloading your machine.

```python
>>> import eland as ed
>>>
# Connect to 'flights' index via localhost Elasticsearch node
>>> df = ed.DataFrame('http://localhost:9200', 'flights')

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

