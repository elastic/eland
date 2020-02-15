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

import distutils.version
import importlib
import types
import warnings

# ----------------------------------------------------------------------------
# functions largely based / taken from the six module

# Much of the code in this module comes from pandas.
# The license for this library can be found NOTICE.txt and the code can be
# https://raw.githubusercontent.com/pandas-dev/pandas/v1.0.1/pandas/compat/_optional.py

VERSIONS = {
    "xgboost": "0.90",
    "sklearn": "0.22.1"
}

# Update install.rst when updating versions!

message = (
    "Missing optional dependency '{name}'. {extra} "
    "Use pip or conda to install {name}."
)
version_message = (
    "Eland requires version '{minimum_version}' or newer of '{name}' "
    "(version '{actual_version}' currently installed). "
    "Use pip or conda to update {name}."
)


def _get_version(module: types.ModuleType) -> str:
    version = getattr(module, "__version__", None)
    if version is None:
        # xlrd uses a capitalized attribute name
        version = getattr(module, "__VERSION__", None)

    if version is None:
        raise ImportError("Can't determine version for {}".format(module.__name__))
    return version


def import_optional_dependency(
        name: str, extra: str = "", raise_on_missing: bool = True, on_version: str = "raise"
):
    """
    Import an optional dependency.

    By default, if a dependency is missing an ImportError with a nice
    message will be raised. If a dependency is present, but too old,
    we raise.

    Parameters
    ----------
    name : str
        The module name. This should be top-level only, so that the
        version may be checked.
    extra : str
        Additional text to include in the ImportError message.
    raise_on_missing : bool, default True
        Whether to raise if the optional dependency is not found.
        When False and the module is not present, None is returned.
    on_version : str {'raise', 'warn'}
        What to do when a dependency's version is too old.

        * raise : Raise an ImportError
        * warn : Warn that the version is too old. Returns None
        * ignore: Return the module, even if the version is too old.
          It's expected that users validate the version locally when
          using ``on_version="ignore"`` (see. ``io/html.py``)

    Returns
    -------
    maybe_module : Optional[ModuleType]
        The imported module, when found and the version is correct.
        None is returned when the package is not found and `raise_on_missing`
        is False, or when the package's version is too old and `on_version`
        is ``'warn'``.
    """
    try:
        module = importlib.import_module(name)
    except ImportError:
        if raise_on_missing:
            raise ImportError(message.format(name=name, extra=extra)) from None
        else:
            return None

    minimum_version = VERSIONS.get(name)
    if minimum_version:
        version = _get_version(module)
        if distutils.version.LooseVersion(version) < minimum_version:
            assert on_version in {"warn", "raise", "ignore"}
            msg = version_message.format(
                minimum_version=minimum_version, name=name, actual_version=version
            )
            if on_version == "warn":
                warnings.warn(msg, UserWarning)
                return None
            elif on_version == "raise":
                raise ImportError(msg)

    return module
