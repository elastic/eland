# Copyright 2020 Elasticsearch BV
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class RandomScore:
    def __init__(self, query):

        q = {"match_all": {}}
        if not query.empty():
            q = query.build()

        self._score = {"function_score": {"query": q, "random_score": {}}}

    def empty(self):
        if self._score is None:
            return True
        return False

    def build(self):
        return self._score
