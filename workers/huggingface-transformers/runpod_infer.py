'''
RunPod | Transformer | Handler
'''
import argparse

import torch
import runpod
from runpod.serverless.utils.rp_validator import validate
from transformers import (GPTNeoForCausalLM, GPT2Tokenizer, GPTNeoXForCausalLM,
                          GPTNeoXTokenizerFast, GPTJForCausalLM, AutoTokenizer, AutoModelForCausalLM)


torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_SCHEMA = {
    'prompt': {
        'type': str,
        'required': True
    },
    'do_sample': {
        'type': bool,
        'required': False,
        'default': True,
        'description': '''
            Enables decoding strategies such as multinomial sampling,
            beam-search multinomial sampling, Top-K sampling and Top-p sampling.
            All these strategies select the next token from the probability distribution
            over the entire vocabulary with various strategy-specific adjustments.
        '''
    },
    'max_length': {
        'type': int,
        'required': False,
        'default': 100
    },
    'temperature': {
        'type': float,
        'required': False,
        'default': 0.9
    }
}


def generator(job):
    '''
    Run the job input to generate text output.
    '''
    # Validate the input
    val_input = validate(job['input'], INPUT_SCHEMA)
    if 'errors' in val_input:
        return {"error": val_input['errors']}
    val_input = val_input['validated_input']

    input_ids = tokenizer(val_input['prompt'], return_tensors="pt").input_ids.to(device)

    gen_tokens = model.generate(
        input_ids,
        do_sample=val_input['do_sample'],
        temperature=val_input['temperature'],
        max_length=val_input['max_length'],
    ).to(device)

    gen_text = tokenizer.batch_decode(gen_tokens)[0]

    return gen_text


# ---------------------------------------------------------------------------- #
#                                Parse Arguments                               #
# ---------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model_name", type=str,
                    default="gpt-neo-1.3B", help="URL of the model to download.")


if __name__ == "__main__":
    args = parser.parse_args()

    # --------------------------------- Neo 1.3B --------------------------------- #
    if args.model_name == 'gpt-neo-1.3B':
        model = GPTNeoForCausalLM.from_pretrained(
            "EleutherAI/gpt-neo-1.3B", local_files_only=True).to(device)
        tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B", local_files_only=True)

    elif args.model_name == 'gpt-neo-2.7B':
        model = GPTNeoForCausalLM.from_pretrained(
            "EleutherAI/gpt-neo-2.7B", local_files_only=True, torch_dtype=torch.float16).to(device)
        tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B", local_files_only=True)

    elif args.model_name == 'gpt-neox-20b':
        model = GPTNeoXForCausalLM.from_pretrained(
            "EleutherAI/gpt-neox-20b", local_files_only=True).half().to(device)
        tokenizer = GPTNeoXTokenizerFast.from_pretrained(
            "EleutherAI/gpt-neox-20b", local_files_only=True)

    elif args.model_name == 'pygmalion-6b':
        model = AutoModelForCausalLM.from_pretrained(
            "PygmalionAI/pygmalion-6b", local_files_only=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            "PygmalionAI/pygmalion-6b", local_files_only=True)

    elif args.model_name == 'gpt-j-6b':
        model = GPTJForCausalLM.from_pretrained(
            "EleutherAI/gpt-j-6B", local_files_only=True, revision="float16",
            torch_dtype=torch.float16).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/gpt-j-6B", local_files_only=True)

    runpod.serverless.start({"handler": generator})
