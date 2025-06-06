#  Licensed to Elasticsearch B.V. under one or more contributor
#  license agreements. See the NOTICE file distributed with
#  this work for additional information regarding copyright
#  ownership. Elasticsearch B.V. licenses this file to you under
#  the Apache License, Version 2.0 (the "License"); you may
#  not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
# 	http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the License is distributed on an
#  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
#  specific language governing permissions and limitations
#  under the License.

import os
from pathlib import Path

import nox

BASE_DIR = Path(__file__).parent
SOURCE_FILES = ("setup.py", "noxfile.py", "eland/", "docs/", "utils/", "tests/")

# Whenever type-hints are completed on a file it should
# be added here so that this file will continue to be checked
# by mypy. Errors from other files are ignored.
TYPED_FILES = (
    "eland/actions.py",
    "eland/arithmetics.py",
    "eland/common.py",
    "eland/etl.py",
    "eland/filter.py",
    "eland/index.py",
    "eland/query.py",
    "eland/tasks.py",
    "eland/utils.py",
    "eland/groupby.py",
    "eland/operations.py",
    "eland/ndframe.py",
    "eland/ml/__init__.py",
    "eland/ml/_optional.py",
    "eland/ml/_model_serializer.py",
    "eland/ml/ml_model.py",
    "eland/ml/pytorch/__init__.py",
    "eland/ml/pytorch/_pytorch_model.py",
    "eland/ml/pytorch/transformers.py",
    "eland/ml/transformers/__init__.py",
    "eland/ml/transformers/base.py",
    "eland/ml/transformers/lightgbm.py",
    "eland/ml/transformers/sklearn.py",
    "eland/ml/transformers/xgboost.py",
    "eland/plotting/_matplotlib/__init__.py",
)


@nox.session(reuse_venv=True, python="3.11")
def format(session):
    session.install("black ~= 25.0", "isort", "flynt")
    session.run("python", "utils/license-headers.py", "fix", *SOURCE_FILES)
    session.run("flynt", *SOURCE_FILES)
    session.run("black", "--target-version=py39", *SOURCE_FILES)
    session.run("isort", "--profile=black", *SOURCE_FILES)
    lint(session)


@nox.session(reuse_venv=True, python="3.11")
def lint(session):
    # Install numpy to use its mypy plugin
    # https://numpy.org/devdocs/reference/typing.html#mypy-plugin
    session.install("black ~= 25.0", "flake8", "mypy", "isort", "numpy")
    session.install(".")
    session.run("python", "utils/license-headers.py", "check", *SOURCE_FILES)
    session.run("black", "--check", "--target-version=py39", *SOURCE_FILES)
    session.run("isort", "--check", "--profile=black", *SOURCE_FILES)
    session.run("flake8", "--extend-ignore=E203,E402,E501,E704,E712", *SOURCE_FILES)

    # TODO: When all files are typed we can change this to .run("mypy", "--strict", "eland/")
    stdout = session.run(
        "mypy",
        "--show-error-codes",
        "--strict",
        *TYPED_FILES,
        success_codes=(0, 1),
        silent=True,
    )

    errors = []
    for line in stdout.splitlines():
        filepath = line.partition(":")[0]
        if filepath in TYPED_FILES:
            errors.append(line)
    if errors:
        session.error("\n" + "\n".join(sorted(set(errors))))


@nox.session(python=["3.9", "3.10", "3.11", "3.12"])
@nox.parametrize("pandas_version", ["1.5.0", "2.2.3"])
def test(session, pandas_version: str):
    session.install("-r", "requirements-dev.txt")
    session.install(".")
    session.run("python", "-m", "pip", "install", f"pandas~={pandas_version}")
    session.run("python", "-m", "tests.setup_tests")

    pytest_args = (
        "python",
        "-m",
        "pytest",
        "-ra",
        "--tb=native",
        "--cov-report=term-missing",
        "--cov=eland/",
        "--cov-config=setup.cfg",
        "--doctest-modules",
        "--nbval",
    )

    session.run(
        *pytest_args,
        *(session.posargs or ("eland/", "tests/")),
    )

    # Only run during default test execution
    if not session.posargs:
        session.run(
            "python",
            "-m",
            "pip",
            "uninstall",
            "--yes",
            "scikit-learn",
            "xgboost",
            "lightgbm",
        )
        session.run("pytest", "tests/ml/")


@nox.session(reuse_venv=True)
def docs(session):
    # Run this so users get an error if they don't have Pandoc installed.
    session.run("pandoc", "--version", external=True)

    session.install(".")
    session.install("-r", "docs/requirements-docs.txt")

    # See if we have an Elasticsearch cluster active
    # to rebuild the Jupyter notebooks with.
    es_active = False
    try:
        from elasticsearch import ConnectionError, Elasticsearch

        try:
            es = Elasticsearch("http://localhost:9200")
            es.info()
            if not es.indices.exists(index="flights"):
                session.run("python", "-m", "tests.setup_tests")
            es_active = True
        except ConnectionError:
            pass
    except ImportError:
        pass

    # Rebuild all the example notebooks inplace
    if es_active:
        session.install("jupyter-client", "ipykernel")
        for filename in os.listdir(BASE_DIR / "docs/sphinx/examples"):
            if (
                filename.endswith(".ipynb")
                and filename != "introduction_to_eland_webinar.ipynb"
            ):
                session.run(
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "notebook",
                    "--inplace",
                    "--execute",
                    str(BASE_DIR / "docs/sphinx/examples" / filename),
                )

    session.cd("docs")
    session.run("make", "clean", external=True)
    session.run("make", "html", external=True)
