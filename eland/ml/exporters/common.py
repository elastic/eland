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

import eland


class ModelDefinitionKeyError(Exception):
    """
    This exception is raised when a key is not found in the model definition.

    Attributes:
        missed_key (str): The key that was not found in the model definition.
        available_keys (List[str]): The list of keys that are available in the model definition.

    Examples:
        model_definition = {"key1": "value1", "key2": "value2"}
        try:
            model_definition["key3"]
        except KeyError as ex:
            raise ModelDefinitionKeyError(ex) from ex
    """

    def __init__(self, ex: KeyError):
        self.missed_key = ex.args[0]

    def __str__(self):
        return (
            f'Key "{self.missed_key}" is not available. '
            + "The model definition may have changed. "
            + "Make sure you are using an Elasticsearch version compatible "
            + f"with Eland {eland.__version__}."
        )
