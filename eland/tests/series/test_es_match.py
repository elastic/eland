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

# File called _pytest for PyCharm compatability

from eland.tests.common import TestData


class TestEsMatch(TestData):
    def test_match(self):
        df = self.ed_ecommerce()

        categories = list(df[df.category.es_match("Men's")].category.to_pandas())
        assert len(categories) > 0
        assert all(any("Men's" in y for y in x) for x in categories)

    def test_must_not_match(self):
        df = self.ed_ecommerce()

        categories = list(
            df[
                ~df.category.es_match("Men's") & df.category.es_match("Women's")
            ].category.to_pandas()
        )
        assert len(categories) > 0
        assert all(all("Men's" not in y for y in x) for x in categories)
        assert all(any("Women's" in y for y in x) for x in categories)
