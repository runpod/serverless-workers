'''
RunPod | Fine-Tuner | rp_handler.py

Handler for the RunPod Fine-Tuner.
This file is responsible for handling the RunPod request and returning the response.
'''

import os
import zipfile

# Import torch and related libraries
import torch
import bitsandbytes as bnb

# Import Hugging Face transformers and related libraries
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig, GPTJForCausalLM,
)
import transformers

# Import datasets
from datasets import load_dataset

# Import RunPod and related libraries
import runpod
from runpod.serverless.utils import (
    rp_validator, download_files_from_urls, upload_file_to_bucket,
)

# Import PEFT and related libraries
from peft import (
    prepare_model_for_int8_training, LoraConfig, get_peft_model,
)

# Import local modules
from rp_schemas import INPUT_SCHEMA

# Check for GPU availability
torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    '''
    Handler for the RunPod Fine-Tuner.
    '''
    job_input = job['input']

    validate_input = rp_validator.validate(job_input, INPUT_SCHEMA)
    if "error" in validate_input:
        return validate_input
    job_input = validate_input['validated_input']

    # Download the dataset
    dataset_file = download_files_from_urls(job['id'], job_input['dataset_url'])[0]

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        job_input['base_model'], add_eos_token=job_input['add_eos_token'])
    model = GPTJForCausalLM.from_pretrained(
        job_input['base_model'], load_in_8bit=job_input['load_in_8bit'], device_map="auto").to(device)

    model = prepare_model_for_int8_training(
        model, use_gradient_checkpointing=job_input['use_gradient_checkpointing']).to(device)

    # Configure Lora
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

    # Load and preprocess dataset
    data = load_dataset('json', data_files=dataset_file)
    data = data.shuffle().map(
        lambda data_point: tokenizer(
            generate_prompt(data_point),
            truncation=True,
            max_length=job_input['cutoff_length'],
            padding='max_length',
        )
    )

    # Configure trainer
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

    # Train the model
    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=False)

    # Save and zip the model
    model.save_pretrained("lora-dolly-finetuned")
    model_path = os.path.join("lora-dolly-finetuned", "model.zip")
    with zipfile.ZipFile(model_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk("lora-dolly-finetuned"):
            for file in files:
                zipf.write(os.path.join(root, file))

    # Upload the model
    uploaded_url = upload_file_to_bucket(
        file_name=f"{job['id']}.zip",
        file_location=model_path
    )

    return {
        'tunned_model_url': uploaded_url,
    }


runpod.serverless.start({"handler": handler})
