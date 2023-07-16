from typing import Union, List

import torch

from .mask_tokens import MaskTokens
from text_utils.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def tokenize(texts: Union[str, List[str]], context_length: int = 77, return_length: bool = False,
             mask_type=None):
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]

    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            all_tokens[i] = [tokens[0]] + tokens[1:context_length - 1] + [tokens[-1]]
        all_tokens[i] = torch.Tensor(all_tokens[i]).long()

    if mask_type is not None:
        mask_token = _tokenizer.encoder["<|mask|>"]
        special_tokens = [sot_token, eot_token, mask_token]
        masked_tokens = [
            MaskTokens(tokens, mask_type=mask_type, mask_token=mask_token, special_tokens=special_tokens,
                       tokenizer_length=len(_tokenizer.encoder)) for tokens in all_tokens]
        all_tokens = [item[0] for item in masked_tokens]
        all_labels = [item[1] for item in masked_tokens]

    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    labels = torch.ones(len(all_tokens), context_length, dtype=torch.long) * -100
    token_lengths = torch.ones(len(all_tokens), dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        result[i, :len(tokens)] = tokens
        token_lengths[i] = min(len(tokens), context_length)
        if mask_type is not None:
            labels[i, :len(tokens)] = all_labels[i]

    if mask_type:
        # print(result[0], labels[0], '<< masking', flush=True)
        return result, labels
    if return_length:
        return result, token_lengths
    else:
        return result