'''
RunPod | Fine-Tuner | rp_handler.py

Handler for the RunPod Fine-Tuner.
This file is responsible for handling the RunPod request and returning the response.
'''

import os

import bitsandbytes as bnb
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig, GPTJForCausalLM
import runpod
from runpod.serverless.utils import rp_validator, download_files_from_urls

from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model

from rp_schemas import INPUT_SCHEMA


def generate_prompt(data_point):
    '''
    Generates a prompt for the model to complete.
    '''
    if data_point['instruction']:
        return f"""
                Below is an instruction that describes a task, paired with an input that provides further context.
                Write a response that appropriately completes the request.

                ### Instruction:
                {data_point["instruction"]}

                ### Input:
                {data_point["input"]}

                ### Response:
                {data_point["output"]}
            """
    else:
        return f"""
                Below is an instruction that describes a task. Write a response that appropriately completes the request.

                ### Instruction:
                {data_point["instruction"]}

                ### Response:
                {data_point["output"]}
            """


def handler(job):
    job_input = job['input']

    validate_input = rp_validator.validate(job_input, INPUT_SCHEMA)
    if "error" in validate_input:
        return validate_input
    job_input = validate_input['validated_input']

    # Download the dataset
    dataset_file = download_files_from_urls(job['id'], job_input['dataset_url'])[0]

    tokenizer = AutoTokenizer.from_pretrained(
        job_input['base_model'], add_eos_token=job_input['add_eos_token'])
    model = GPTJForCausalLM.from_pretrained(
        job_input['base_model'], load_in_8bit=job_input['load_in_8bit'], device_map="auto")

    model = prepare_model_for_int8_training(
        model, use_gradient_checkpointing=job_input['use_gradient_checkpointing'])

    config = LoraConfig(
        r=job_input['lora_rank'],
        lora_alpha=job_input['lora_alpha'],
        target_modules=["q_proj", "v_proj"],
        lora_dropout=job_input['lora_dropout'],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    tokenizer.pad_token_id = 0

    data = load_dataset('json', data_files=dataset_file)

    data = data.shuffle().map(
        lambda data_point: tokenizer(
            generate_prompt(data_point),
            truncation=True,
            max_length=job_input['cutoff_length'],
            padding='max_length',
        )
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=data['train'],
        args=transformers.TrainingArguments(
            per_device_train_batch_size=job_input['micro_batch_size'],
            gradient_accumulation_steps=job_input['batch_size'] // job_input['micro_batch_size'],
            warmup_steps=100,
            num_train_epochs=job_input['num_epochs'],
            learning_rate=job_input['learning_rate'],
            fp16=True,
            logging_steps=1,
            output_dir="lora-dolly",
            save_total_limit=3,
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=False)

    model.save_pretrained("lora-dolly-finetuned")

    model_path = os.path.join("lora-dolly-finetuned", "pytorch_model.bin")

    return f"Model saved at {model_path}"


runpod.serverless.start({"handler": handler})
