'''
RunPod | Transformer | Model Fetcher
'''

import argparse

import torch
from transformers import (GPTNeoForCausalLM, GPT2Tokenizer, GPTNeoXForCausalLM,
                          GPTNeoXTokenizerFast, GPTJForCausalLM, AutoTokenizer, AutoModelForCausalLM)


def download_model(model_name):

    # --------------------------------- Neo 1.3B --------------------------------- #
    if model_name == 'gpt-neo-1.3B':
        GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
        GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

    # --------------------------------- Neo 2.7B --------------------------------- #
    elif model_name == 'gpt-neo-2.7B':
        GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B", torch_dtype=torch.float16)
        GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")

    # ----------------------------------- NeoX ----------------------------------- #
    elif model_name == 'gpt-neox-20b':
        GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b").half()
        GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")

    # --------------------------------- Pygmalion -------------------------------- #
    elif model_name == 'pygmalion-6b':
        AutoModelForCausalLM.from_pretrained("PygmalionAI/pygmalion-6b")
        AutoTokenizer.from_pretrained("PygmalionAI/pygmalion-6b")

    # ----------------------------------- GPT-J ----------------------------------- #
    elif model_name == 'gpt-j-6b':
        GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16",
                                        torch_dtype=torch.float16)
        AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")


# ---------------------------------------------------------------------------- #
#                                Parse Arguments                               #
# ---------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model_name", type=str,
                    default="gpt-neo-1.3B", help="URL of the model to download.")


if __name__ == "__main__":
    args = parser.parse_args()
    download_model(args.model_name)
