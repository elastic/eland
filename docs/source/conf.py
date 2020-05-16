# Licensed to Elasticsearch B.V under one or more agreements.
# Elasticsearch B.V licenses this file to you under the Apache 2.0 License.
# See the LICENSE file in the project root for more information

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import datetime

sys.path.insert(0, os.path.abspath("../sphinxext"))
sys.path.extend(
    [
        # numpy standard doc extensions
        os.path.join(os.path.dirname(__file__), "..", "../..", "sphinxext")
    ]
)

# -- Project information -----------------------------------------------------

project = "eland"
copyright = f"{datetime.date.today().year}, Elasticsearch BV"

# The full version, including alpha/beta/rc tags
import eland

version = str(eland._version.__version__)

release = version

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "numpydoc",
    "matplotlib.sphinxext.plot_directive",
    "sphinx.ext.todo",
    "nbsphinx",
]

doctest_global_setup = """
try:
    import eland as ed
except ImportError:
    ed = None
try:
    import pandas as pd
except ImportError:
    pd = None
"""

extlinks = {
    "pandas_api_docs": (
        "https://pandas.pydata.org/pandas-docs/stable/reference/api/%s.html",
        "",
    ),
    "pandas_user_guide": (
        "https://pandas.pydata.org/pandas-docs/stable/user_guide/%s.html",
        "Pandas User Guide/",
    ),
    "es_api_docs": (
        "https://www.elastic.co/guide/en/elasticsearch/reference/current/%s.html",
        "",
    ),
}

numpydoc_attributes_as_param_list = False
numpydoc_show_class_members = False

# matplotlib plot directive
plot_include_source = True
plot_formats = [("png", 90)]
plot_html_show_formats = False
plot_html_show_source_link = False
plot_pre_code = """import numpy as np
import eland as ed"""

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["**.ipynb_checkpoints"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "external_links": [],
    "github_url": "https://github.com/elastic/eland",
    "twitter_url": "https://twitter.com/elastic",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

html_logo = "logo/eland.png"
html_favicon = "logo/eland_favicon.png"

master_doc = "index"
