import os
from pathlib import Path
import nox
import elasticsearch


BASE_DIR = Path(__file__).parent
SOURCE_FILES = (
    "setup.py",
    "noxfile.py",
    "eland/",
    "docs/",
)


@nox.session(reuse_venv=True)
def blacken(session):
    session.install("black")
    session.run("black", "--target-version=py36", *SOURCE_FILES)
    lint(session)


@nox.session(reuse_venv=True)
def lint(session):
    session.install("black", "flake8")
    session.run("black", "--check", "--target-version=py36", *SOURCE_FILES)
    session.run("flake8", "--ignore=E501,W503,E402,E712", *SOURCE_FILES)


@nox.session(python=["3.6", "3.7", "3.8"])
def test(session):
    session.install("-r", "requirements-dev.txt")
    session.run("python", "-m", "eland.tests.setup_tests")
    session.run("pytest", "--doctest-modules", *(session.posargs or ("eland/",)))


@nox.session(reuse_venv=True)
def docs(session):
    # Run this so users get an error if they don't have Pandoc installed.
    session.run("pandoc", "--version", external=True)

    session.install("-r", "docs/requirements-docs.txt")
    session.run("python", "setup.py", "install")

    # See if we have an Elasticsearch cluster active
    # to rebuild the Jupyter notebooks with.
    try:
        es = elasticsearch.Elasticsearch("localhost:9200")
        es.info()
        if not es.indices.exists("flights"):
            session.run("python", "-m", "eland.tests.setup_tests")
        es_active = True
    except Exception:
        es_active = False

    # Rebuild all the example notebooks inplace
    if es_active:
        session.install("jupyter-client", "ipykernel")
        for filename in os.listdir(BASE_DIR / "docs/source/examples"):
            if filename.endswith(".ipynb"):
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
