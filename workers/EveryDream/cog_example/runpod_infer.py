'''
RunPod | Endpoint | EveryDream | runpod_infer.py

This is the inference start for EveryDream.
It will take the input from the API call, and prepare it for the model.
It will then call the model, and return the results.
'''

import runpod
from runpod.serverless.utils import rp_validator, rp_download

TRAIN_SCHEMA = {
    'amp': {
        'type': bool,
        'required': False,
        'default': True
    },
    'batch_size': {
        'type': int,
        'required': False,
        'default': 2
    },
    'ckpt_every_n_minutes': {
        'type': int,
        'required': False,
        'default': 20
    },
    'clip_grad_norm': {
        'type': float,
        'required': False,
        'default': None
    },
    'clip_skip': {
        'type': int,
        'required': False,
        'default': 0,
        'constraints': lambda clip_skip: clip_skip in [0, 1, 2, 3, 4]
    },
    'cond_dropout': {
        'type': float,
        'required': False,
        'default': 0.04,
        'constraints': lambda cond_dropout: 0 <= cond_dropout <= 1
    },
    'data_url': {
        'type': str,
        'required': True
    },
    'disable_textenc_training': {
        'type': bool,
        'required': False,
        'default': False
    },
    'disable_unet_training': {
        'type': bool,
        'required': False,
        'default': False
    },
    'disable_xformers': {
        'type': bool,
        'required': False,
        'default': False
    },
    'flip_p': {
        'type': float,
        'required': False,
        'default': 0.0,
        'constraints': lambda flip_p: 0 <= flip_p <= 1
    },
    'gradient_checkpointing': {
        'type': bool,
        'required': False,
        'default': False
    },
    'grad_accum': {
        'type': int,
        'required': False,
        'default': 1
    },
    'hf_repo_subfolder': {
        'type': str,
        'required': False,
        'default': None
    },
    'lr': {
        'type': float,
        'required': False,
        'default': None
    },
    'lr_decay_steps': {
        'type': int,
        'required': False,
        'default': None
    },
    'lr_scheduler': {
        'type': str,
        'required': False,
        'default': "constant",
        'constraints': lambda lrs: lrs in ["constant", "linear", "cosine", "polynomial"]
    },
    'lr_warmup_steps': {
        'type': int,
        'required': False,
        'default': None
    },
    'max_epochs': {
        'type': int,
        'required': False,
        'default': 300
    },
    'resolution': {
        'type': int,
        'required': False,
        'default': 512,
        'constraints': lambda resolution: resolution in [256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152]
    },
    'resume_ckpt_url': {
        'type': str,
        'required': False,
        'default': "sd_v1-5_vae.ckpt"
    },
    'sample_prompts': {
        'type': list,
        'required': False,
        'default': None
    },
    'sample_steps': {
        'type': int,
        'required': False,
        'default': 250
    },
    'save_full_precision': {
        'type': bool,
        'required': False,
        'default': False
    },
    'save_optimizer': {
        'type': bool,
        'required': False,
        'default': False
    },
    'scale_lr': {
        'type': bool,
        'required': False,
        'default': False
    },
    'seed': {
        'type': int,
        'required': False,
        'default': 555
    },
    'shuffle_tags': {
        'type': bool,
        'required': False,
        'default': False
    },
    'useadam8bit': {
        'type': bool,
        'required': False,
        'default': True
    },
    'rated_dataset': {
        'type': bool,
        'required': False,
        'default': False
    },
    'rated_dataset_target_dropout_percent': {
        'type': int,
        'required': False,
        'default': 50,
        'constraints': lambda dataset_dropout_percent: 0 <= dataset_dropout_percent <= 100
    }
}

S3_SCHEMA = {
    'accessId': {
        'type': str,
        'required': True
    },
    'accessSecret': {
        'type': str,
        'required': True
    },
    'bucketName': {
        'type': str,
        'required': True
    },
    'endpointUrl': {
        'type': str,
        'required': True
    }
}


def everydream_runner(job):
    '''
    Takes in raw data from the API call, and prepares it for the model.
    Passes the data to the model to get the results.
    Prepares the resulting output to be returned.
    '''
    job_input = job['input']

    # -------------------------------- Validation -------------------------------- #
    # Validate the training input
    validated_train_input = rp_validator(job_input['train'], TRAIN_SCHEMA)
    if 'errors' in validated_train_input:
        return {"error": validated_train_input['errors']}
    train_input = validated_train_input['validated_input']

    # Validate the S3 config, if provided
    if job['s3Config']:
        validated_s3_config = rp_validator(job['s3Config'], S3_SCHEMA)
        if 'errors' in validated_s3_config:
            return {"error": validated_s3_config['errors']}
        s3_config = validated_s3_config['validated_input']

    # --------------------------------- Downloads -------------------------------- #
    # Convert 'data_url' to 'data_root'
    downloaded_input = rp_download.file(train_input['data_url'])
    if downloaded_input['type'] != "zip":
        return {"error": "data_url must be a zip file"}

    train_input['data_root'] = downloaded_input['extracted_path']

    # Download the resume checkpoint, if provided
    if train_input['resume_ckpt_url'] != "sd_v1-5_vae.ckpt":

        # Check if the URL is from huggingface.co, if so, grab the model repo id.

        resume_ckpt = rp_download.file(train_input['resume_ckpt_url'])

        train_input['resume_ckpt'] = resume_ckpt['downloaded_path']

    # ------------------------------- Format Inputs ------------------------------ #

    # ------------------------------- Set Defaults ------------------------------- #
    # Set default values for optional parameters
    train_input['project_name'] = job['id']
