# What is it?

eland is a elasticsearch client Python package to analyse, explore and manipulate data that resides in elasticsearch. Where possible the package uses existing Python APIs and data structures to make it easy to switch between Numpy, Pandas, Scikit-learn to their elasticsearch powered equivalents. In general, the data resides in elasticsearch and not in memory, which allows eland to access large datasets stored in elasticsearch.

For example, to explore data in a large elasticsearch index, simply create an eland DataFrame from an elasticsearch index pattern, and explore using an API that mirrors a subset of the pandas.DataFrame API: 

```
>>> import eland as ed

>>> df = ed.read_es(url='http://localhost:9200', index='reviews') 

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

## Why eland?

Naming is difficult, but as we had to call it something:

* eland = elastic and data
* eland = 'Elk/Moose' in Dutch (Alces alces)
* Elandsgracht = Amsterdam street near Elastic's Amsterdam office where historically hides from, among others, Elk were worked

