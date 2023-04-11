# ---------------------------------------------------------------------------- #
#                                    Schemas                                   #
# ---------------------------------------------------------------------------- #
TRAIN_SCHEMA = {
    'data_url': {
        'type': str,
        'required': True
    },
    'hf_model': {
        'type': str,
        'required': False,
        'default': None
    },
    'hf_token': {
        'type': str,
        'required': False,
        'default': None
    },
    'ckpt_link': {
        'type': str,
        'required': False,
        'default': None
    },
    'concept_name': {
        'type': str,
        'required': False,
        'default': None
    },
    'offset_noise': {
        'type': bool,
        'required': False,
        'default': False
    },
    'pndm_scheduler': {
        'type': bool,
        'required': False,
        'default': False
    },
    # Text Encoder Training Parameters
    'text_steps': {
        'type': int,
        'required': False,
        'default': 350
    },
    'text_seed': {
        'type': int,
        'required': False,
        'default': 555
    },
    'text_batch_size': {
        'type': int,
        'required': False,
        'default': 1
    },
    'text_resolution': {
        'type': int,
        'required': False,
        'default': 512
    },
    'text_learning_rate': {
        'type': float,
        'required': False,
        'default': 1e-6
    },
    'text_lr_scheduler': {
        'type': str,
        'required': False,
        'default': 'linear',
        'constraints': lambda scheduler: scheduler in ['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup']
    },
    'text_8_bit_adam': {
        'type': bool,
        'required': False,
        'default': False
    },
    # UNet Training Parameters
    'unet_seed': {
        'type': int,
        'required': False,
        'default': 555
    },
    'unet_batch_size': {
        'type': int,
        'required': False,
        'default': 1
    },
    'unet_resolution': {
        'type': int,
        'required': False,
        'default': 256
    },
    'unet_epochs': {
        'type': int,
        'required': False,
        'default': 150
    },
    'unet_learning_rate': {
        'type': float,
        'required': False,
        'default': 2e-6
    },
    'unet_lr_scheduler': {
        'type': str,
        'required': False,
        'default': 'linear',
        'constraints': lambda scheduler: scheduler in ['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup']
    },
    'unet_8_bit_adam': {
        'type': bool,
        'required': False,
        'default': False
    }
}

INFERENCE_SCHEMA = {
    'enable_hr': {
        'type': bool,
        'required': False,
        'default': False
    },
    'denoising_strength': {
        'type': float,
        'required': False,
        'default': 0
    },
    'firstphase_width': {
        'type': int,
        'required': False,
        'default': 0
    },
    'firstphase_height': {
        'type': int,
        'required': False,
        'default': 0
    },
    'hr_scale': {
        'type': int,
        'required': False,
        'default': 2
    },
    'hr_upscaler': {
        'type': str,
        'required': False,
        'default': None
    },
    'hr_second_pass_steps': {
        'type': int,
        'required': False,
        'default': 0
    },
    'hr_resize_x': {
        'type': int,
        'required': False,
        'default': 0
    },
    'hr_resize_y': {
        'type': int,
        'required': False,
        'default': 0
    },
    'prompt': {
        'type': str,
        'required': True
    },
    'styles': {
        'type': list,
        'required': False,
        'default': []
    },
    'seed': {
        'type': int,
        'required': False,
        'default': -1
    },
    'subseed': {
        'type': int,
        'required': False,
        'default': -1
    },
    'subseed_strength': {
        'type': int,
        'required': False,
        'default': 0
    },
    'seed_resize_from_h': {
        'type': int,
        'required': False,
        'default': -1
    },
    'seed_resize_from_w': {
        'type': int,
        'required': False,
        'default': -1
    },
    'sampler_name': {
        'type': str,
        'required': False,
        'default': 'Euler'
    },
    'batch_size': {
        'type': int,
        'required': False,
        'default': 1
    },
    'n_iter': {
        'type': int,
        'required': False,
        'default': 1
    },
    'steps': {
        'type': int,
        'required': False,
        'default': 50
    },
    'cfg_scale': {
        'type': float,
        'required': False,
        'default': 7.0
    },
    'width': {
        'type': int,
        'required': False,
        'default': 512
    },
    'height': {
        'type': int,
        'required': False,
        'default': 512
    },
    'restore_faces': {
        'type': bool,
        'required': False,
        'default': False
    },
    'tiling': {
        'type': bool,
        'required': False,
        'default': False
    },
    'negative_prompt': {
        'type': str,
        'required': False,
        'default': None
    },
    'eta': {
        'type': int,
        'required': False,
        'default': None
    },
    's_churn': {
        'type': int,
        'required': False,
        'default': 0
    },
    's_tmax': {
        'type': int,
        'required': False,
        'default': None
    },
    's_tmin': {
        'type': int,
        'required': False,
        'default': 0
    },
    's_noise': {
        'type': int,
        'required': False,
        'default': 1
    },
    'sampler_index': {
        'type': str,
        'required': False,
        'default': 'Euler',
    },
    'script_name': {
        'type': str,
        'required': False,
        'default': None
    },
    'passback': {
        'type': str,
        'required': False,
        'default': None
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
