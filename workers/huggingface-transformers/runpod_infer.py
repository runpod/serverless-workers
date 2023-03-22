'''
RunPod | Transformer | Handler
'''
import runpod
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast

model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neo-1.3B")


def generator(job):
    '''
    Run the job input to generate text output.
    '''
    job_input = job['input']

    prompt = job_input['prompt']

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        temperature=0.9,
        max_length=100,
    )
    gen_text = tokenizer.batch_decode(gen_tokens)[0]

    return gen_text


# ---------------------------------------------------------------------------- #
#                                 Start Worker                                 #
# ---------------------------------------------------------------------------- #
runpod.serverless.start({"handler": generator})
