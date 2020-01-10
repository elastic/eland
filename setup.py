#  Copyright 2019 Elasticsearch BV
#
#      Licensed under the Apache License, Version 2.0 (the "License");
#      you may not use this file except in compliance with the License.
#      You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#      Unless required by applicable law or agreed to in writing, software
#      distributed under the License is distributed on an "AS IS" BASIS,
#      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#      See the License for the specific language governing permissions and
#      limitations under the License.

from codecs import open
from os import path

from setuptools import setup

here = path.abspath(path.dirname(__file__))
about = {}
with open(path.join(here, 'eland', '_version.py'), 'r', 'utf-8') as f:
    exec(f.read(), about)

CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "License :: OSI Approved :: Apache Software License",
    "Environment :: Console",
    "Operating System :: OS Independent",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.5",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Cython",
    "Topic :: Scientific/Engineering",
]

LONG_DESCRIPTION="""
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
"""

setup(
    name=about['__title__'],
    version=about['__version__'],
    description=about['__description__'],
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    url=about['__url__'],
    maintainer=about['__maintainer__'],
    maintainer_email=about['__maintainer_email__'],
    license='Apache 2.0',
    classifiers=CLASSIFIERS,
    keywords='elastic eland pandas python',
    packages=['eland'],
    install_requires=[
        'elasticsearch>=7.0.5',
        'pandas==0.25.3',
        'matplotlib'
    ]
)
