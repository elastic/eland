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

from typing import Optional, Union

import transformers

from eland.ml.pytorch.nlp_ml_model import (
    NlpBertJapaneseTokenizationConfig,
    NlpBertTokenizationConfig,
    NlpDebertaV2TokenizationConfig,
    NlpMPNetTokenizationConfig,
    NlpRobertaTokenizationConfig,
    NlpTokenizationConfig,
    NlpXLMRobertaTokenizationConfig,
)

SUPPORTED_TOKENIZERS = (
    transformers.BertTokenizer,
    transformers.BertTokenizerFast,
    transformers.BertJapaneseTokenizer,
    transformers.MPNetTokenizer,
    transformers.MPNetTokenizerFast,
    transformers.DPRContextEncoderTokenizer,
    transformers.DPRContextEncoderTokenizerFast,
    transformers.DPRQuestionEncoderTokenizer,
    transformers.DPRQuestionEncoderTokenizerFast,
    transformers.DistilBertTokenizer,
    transformers.DistilBertTokenizerFast,
    transformers.ElectraTokenizer,
    transformers.ElectraTokenizerFast,
    transformers.MobileBertTokenizer,
    transformers.MobileBertTokenizerFast,
    transformers.RetriBertTokenizer,
    transformers.RetriBertTokenizerFast,
    transformers.RobertaTokenizer,
    transformers.RobertaTokenizerFast,
    transformers.BartTokenizer,
    transformers.BartTokenizerFast,
    transformers.SqueezeBertTokenizer,
    transformers.SqueezeBertTokenizerFast,
    transformers.XLMRobertaTokenizer,
    transformers.XLMRobertaTokenizerFast,
    transformers.DebertaV2Tokenizer,
    transformers.DebertaV2TokenizerFast,
)
SUPPORTED_TOKENIZERS_NAMES = ", ".join(sorted([str(x) for x in SUPPORTED_TOKENIZERS]))


class UnknownModelInputSizeError(Exception):
    pass


def find_max_sequence_length(
    model_id: str,
    tokenizer: Union[
        transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast
    ],
) -> int:
    # Sometimes the max_... values are present but contain
    # a random or very large value.
    REASONABLE_MAX_LENGTH = 8192
    max_len = getattr(tokenizer, "model_max_length", None)
    if max_len is not None and max_len <= REASONABLE_MAX_LENGTH:
        return int(max_len)

    max_sizes = getattr(tokenizer, "max_model_input_sizes", dict())
    max_len = max_sizes.get(model_id)
    if max_len is not None and max_len < REASONABLE_MAX_LENGTH:
        return int(max_len)

    if max_sizes:
        # The model id wasn't found in the max sizes dict but
        # if all the values correspond then take that value
        sizes = {size for size in max_sizes.values()}
        if len(sizes) == 1:
            max_len = sizes.pop()
            if max_len is not None and max_len < REASONABLE_MAX_LENGTH:
                return int(max_len)

    if isinstance(
        tokenizer, (transformers.BertTokenizer, transformers.BertTokenizerFast)
    ):
        return 512

    raise UnknownModelInputSizeError("Cannot determine model max input length")


def create_tokenization_config(
    model_id: str,
    max_model_input_size: Optional[int],
    tokenizer: Union[
        transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast
    ],
) -> NlpTokenizationConfig:
    if max_model_input_size is not None:
        _max_sequence_length = max_model_input_size
    else:
        _max_sequence_length = find_max_sequence_length(model_id, tokenizer)

    if isinstance(
        tokenizer, (transformers.MPNetTokenizer, transformers.MPNetTokenizerFast)
    ):
        return NlpMPNetTokenizationConfig(
            do_lower_case=getattr(tokenizer, "do_lower_case", None),
            max_sequence_length=_max_sequence_length,
        )
    elif isinstance(
        tokenizer,
        (
            transformers.RobertaTokenizer,
            transformers.RobertaTokenizerFast,
            transformers.BartTokenizer,
            transformers.BartTokenizerFast,
        ),
    ):
        return NlpRobertaTokenizationConfig(
            add_prefix_space=getattr(tokenizer, "add_prefix_space", None),
            max_sequence_length=_max_sequence_length,
        )
    elif isinstance(
        tokenizer,
        (transformers.XLMRobertaTokenizer, transformers.XLMRobertaTokenizerFast),
    ):
        return NlpXLMRobertaTokenizationConfig(max_sequence_length=_max_sequence_length)
    elif isinstance(
        tokenizer,
        (transformers.DebertaV2Tokenizer, transformers.DebertaV2TokenizerFast),
    ):
        return NlpDebertaV2TokenizationConfig(
            max_sequence_length=_max_sequence_length,
            do_lower_case=getattr(tokenizer, "do_lower_case", None),
        )
    else:
        japanese_morphological_tokenizers = ["mecab"]
        if (
            hasattr(tokenizer, "word_tokenizer_type")
            and tokenizer.word_tokenizer_type in japanese_morphological_tokenizers
        ):
            return NlpBertJapaneseTokenizationConfig(
                do_lower_case=getattr(tokenizer, "do_lower_case", None),
                max_sequence_length=_max_sequence_length,
            )
        else:
            return NlpBertTokenizationConfig(
                do_lower_case=getattr(tokenizer, "do_lower_case", None),
                max_sequence_length=_max_sequence_length,
            )
