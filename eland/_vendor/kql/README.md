# Vendored: kql

Kibana Query Language (KQL) parser, copied from the `lib/kql/kql` directory of
[elastic/detection-rules](https://github.com/elastic/detection-rules/tree/main/lib/kql).

- Upstream commit: `c3e4e4e75aa8914dd5a74360f9bbe33b186d8b82`
- Upstream package: `detection-rules-kql` 0.1.17
- License: Elastic License 2.0 (see `LICENSE.txt` in this directory)

Local modifications:

- Absolute `from kql.errors import ...` imports were changed to relative
  imports in `parser.py` and `utils.py`, so the package works when imported
  as `eland._vendor.kql`.

The upstream dependencies (`eql`, `lark`) are listed in eland's `setup.py`.
