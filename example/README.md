# Example Walkthrough for eland

This example demonstrate the functionality of `eland` through a walkthrough of a simple analysis of the [Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/online+retail).

To run this example, make sure that you have an elasticsearch cluster running on port 9200 and please install any additional dependencies in addition to `eland`:

```
pip install -r requirements-example.txt
```

Once these requirements are satisfied, load the data using the provided script:

```
python load.py
```

This will create an index called `online-retail` with a mapping defined in `load.py`.