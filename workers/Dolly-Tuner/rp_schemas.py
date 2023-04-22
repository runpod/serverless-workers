'''
RunPod | Fine-Tuner | rp_schemas.py

Contains the schemas for the RunPod Fine-Tuner inputs.
'''

INPUT_SCHEMA = {
    'base_model': {
        'type': str,
        'required': False,
        'default': 'databricks/dolly-v2-12b'
    },
    'dataset_url': {
        'type': str,
        'required': True},
    'micro_batch_size': {
        'type': int,
        'required': False,
        'default': 8
    },
    'batch_size': {
        'type': int,
        'required': False,
        'default': 128
    },
    'num_epochs': {
        'type': int,
        'required': False,
        'default': 2
    },
    'learning_rate': {
        'type': float,
        'required': False,
        'default': 2e-5
    },
    'cutoff_length': {
        'type': int,
        'required': False,
        'default': 256
    },
    'lora_rank': {
        'type': int,
        'required': False,
        'default': 4
    },
    'lora_alpha': {
        'type': float,
        'required': False,
        'default': 16.0
    },
    'lora_dropout': {
        'type': float,
        'required': False,
        'default': 0.05
    },
    'add_eos_token': {
        'type': bool,
        'required': False,
        'default': True
    },
    'load_in_8bit': {
        'type': bool,
        'required': False,
        'default': True
    },
    'use_gradient_checkpointing': {
        'type': bool,
        'required': False,
        'default': True
    },
}
