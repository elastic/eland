#!/bin/sh

python setup.py install

#jupyter nbconvert --to notebook --inplace --execute docs/source/examples/demo_notebook.ipynb
#jupyter nbconvert --to notebook --inplace --execute docs/source/examples/online_retail_analysis.ipynb 

cd docs

make clean
make html

