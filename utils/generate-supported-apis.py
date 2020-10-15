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

"""Script that is used to create the compatibility matrix in the documentation"""

import inspect
import re
from pathlib import Path

import pandas

import eland

api_docs_dir = Path(__file__).absolute().parent.parent / "docs/source/reference/api"
is_supported = []
supported_attr = re.compile(
    r"(?:[a-zA-Z0-9][a-zA-Z0-9_]*|__[a-zA-Z0-9][a-zA-Z0-9_]*__)"
)


def main():
    for prefix, pd_obj, ed_obj in [
        ("ed.DataFrame.", pandas.DataFrame, eland.DataFrame),
        ("ed.Series.", pandas.Series, eland.Series),
    ]:
        total = 0
        supported = 0
        for attr in sorted(dir(pd_obj), key=lambda x: (x.startswith("__"), x.lower())):
            val = getattr(pd_obj, attr)
            if inspect.isclass(val) or inspect.ismodule(val):
                continue

            total += 1
            suffix = ""
            if inspect.ismethod(val) or inspect.isfunction(val):
                suffix = "()"

            if supported_attr.fullmatch(attr):
                supported += hasattr(ed_obj, attr)
                is_supported.append((prefix + attr + suffix, hasattr(ed_obj, attr)))

        print(
            prefix.rstrip("."),
            f"{supported} / {total} ({100.0 * supported / total:.1f}%)",
        )

    column1_width = max([len(attr) + 1 for attr, _ in is_supported])
    row_delimiter = f"+{'-' * (column1_width + 5)}+------------+"

    print(row_delimiter)
    print(f"| Method or Property{' ' * (column1_width - 15)} | Supported? |")
    print(row_delimiter.replace("-", "="))

    for attr, supported in is_supported:
        print(
            f"| ``{attr}``{' ' * (column1_width - len(attr))}|{' **Yes**    ' if supported else ' No         '}|"
        )
        print(row_delimiter)

    for attr, supported in is_supported:
        if supported and "__" not in attr:
            attr = attr.replace("ed.", "eland.").rstrip("()")
            attr_doc_path = api_docs_dir / f"{attr}.rst"
            if not attr_doc_path.exists():
                with attr_doc_path.open(mode="w") as f:
                    f.truncate()
                    f.write(
                        f"""{attr}
{'=' * len(attr)}

.. currentmodule:: eland

.. automethod:: { attr.replace('eland.', '') }
"""
                    )


if __name__ == "__main__":
    main()
