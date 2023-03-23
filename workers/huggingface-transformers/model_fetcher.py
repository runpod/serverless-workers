'''
RunPod | Transformer | Model Fetcher
'''

import argparse

from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM


def download_model(model_name):
    if model_name == 'gpt-neo-1.3B':
        GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B", local_files_only=True)
        GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B", local_files_only=True)
    elif model_name == 'gpt-neox-20b':
        AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", local_files_only=True)
        AutoModelForCausalLM.from_pretrained(
            "EleutherAI/gpt-neox-20b", local_files_only=True)


# ---------------------------------------------------------------------------- #
#                                Parse Arguments                               #
# ---------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model_name", type=str,
                    default="gpt-neo-1.3B", help="URL of the model to download.")


if __name__ == "__main__":
    args = parser.parse_args()
    download_model(args.model_name)
