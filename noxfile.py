import nox


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
