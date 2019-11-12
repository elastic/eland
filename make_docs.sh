#!/bin/sh

python setup.py install

cd docs

make clean
make html

