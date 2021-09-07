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
import subprocess
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
    "eland/ml/transformers/__init__.py",
    "eland/ml/transformers/base.py",
    "eland/ml/transformers/lightgbm.py",
    "eland/ml/transformers/sklearn.py",
    "eland/ml/transformers/xgboost.py",
    "eland/plotting/_matplotlib/__init__.py",
)


@nox.session(reuse_venv=True)
def format(session):
    session.install("black", "isort")
    session.run("python", "utils/license-headers.py", "fix", *SOURCE_FILES)
    session.run("black", "--target-version=py37", *SOURCE_FILES)
    session.run("isort", *SOURCE_FILES)
    lint(session)


@nox.session(reuse_venv=True)
def lint(session):
    # Install numpy to use its mypy plugin
    # https://numpy.org/devdocs/reference/typing.html#mypy-plugin
    session.install("black", "flake8", "mypy", "isort", "numpy")
    session.install("--pre", "elasticsearch")
    session.run("python", "utils/license-headers.py", "check", *SOURCE_FILES)
    session.run("black", "--check", "--target-version=py37", *SOURCE_FILES)
    session.run("isort", "--check", *SOURCE_FILES)
    session.run("flake8", "--ignore=E501,W503,E402,E712,E203", *SOURCE_FILES)

    # TODO: When all files are typed we can change this to .run("mypy", "--strict", "eland/")
    session.log("mypy --strict eland/")
    for typed_file in TYPED_FILES:
        if not os.path.isfile(typed_file):
            session.error(f"The file {typed_file!r} couldn't be found")
        process = subprocess.run(
            ["mypy", "--strict", typed_file],
            env=session.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        # Ensure that mypy itself ran successfully
        assert process.returncode in (0, 1)

        errors = []
        for line in process.stdout.decode().split("\n"):
            filepath = line.partition(":")[0]
            if filepath in TYPED_FILES:
                errors.append(line)
        if errors:
            session.error("\n" + "\n".join(sorted(set(errors))))


@nox.session(python=["3.7", "3.8", "3.9"])
@nox.parametrize("pandas_version", ["1.2.0", "1.3.0"])
def test(session, pandas_version: str):
    session.install("-r", "requirements-dev.txt")
    session.install(".")
    session.run("python", "-m", "pip", "install", f"pandas~={pandas_version}")
    session.run("python", "-m", "tests.setup_tests")
    session.run(
        "python",
        "-m",
        "pytest",
        "--cov-report",
        "term-missing",
        "--cov=eland/",
        "--cov-config=setup.cfg",
        "--doctest-modules",
        "--nbval",
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

    session.install("-r", "docs/requirements-docs.txt")
    session.install(".")

    # See if we have an Elasticsearch cluster active
    # to rebuild the Jupyter notebooks with.
    try:
        import elasticsearch

        es = elasticsearch.Elasticsearch("localhost:9200")
        es.info()
        if not es.indices.exists("flights"):
            session.run("python", "-m", "tests.setup_tests")
        es_active = True
    except Exception:
        es_active = False

    # Rebuild all the example notebooks inplace
    if es_active:
        session.install("jupyter-client", "ipykernel")
        for filename in os.listdir(BASE_DIR / "docs/source/examples"):
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
                    str(BASE_DIR / "docs/source/examples" / filename),
                )

    session.cd("docs")
    session.run("make", "clean", external=True)
    session.run("make", "html", external=True)
