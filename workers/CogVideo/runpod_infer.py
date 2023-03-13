'''
RunPod | Inference
'''

import predict

PREDICT_SCHEMA = {
    'prompt': {
        'type': str,
        'required': True
    },
    'seed': {
        'type': int,
        'required': False,
        'default': None
    },
    'translate': {
        'type': str,
        'required': False,
        'default': True
    },
    'both_stages': {
        'type': bool,
        'required': False,
        'default': True
    },
    'use_guidance': {
        'type': bool,
        'required': False,
        'default': True
    },
    'image_prompt': {
        'type': str,
        'required': False,
        'default': None
    }
}

ARG_SCHEMA = {
    'layout': {
        'type': list,
        'required': False,
        'default': [64, 464, 2064]
    },
    'window_size': {
        'type': int,
        'required': False,
        'default': 10
    },
    'additional_seqlen': {
        'type': int,
        'required': False,
        'default': 2000
    },
    'cogvideo_stage': {
        'type': int,
        'required': False,
        'default': 1
    },
    'do_train': {
        'type': bool,
        'required': False,
        'default': False
    },
    'parallel_size': {
        'type': int,
        'required': False,
        'default': 1
    },
    'guidance_alpha': {
        'type': float,
        'required': False,
        'default': 3.0
    },
    'generate_frame_num': {
        'type': int,
        'required': False,
        'default': 5
    },
    'coglm_temperature': {
        'type': float,
        'required': False,
        'default': 0.89
    },
    'coglm_temperature2': {
        'type': float,
        'required': False,
        'default': 0.89
    },
    'generate_frame_num': {
        'type': int,
        'required': False,
        'default': 5
    },
    'stage1_max_inference_batch_size': {
        'type': int,
        'required': False,
        'default': -1
    },
    'max_inference_batch_size': {
        'type': int,
        'required': False,
        'default': 8
    },
    'top_k': {
        'type': int,
        'required': False,
        'default': 12
    },
    'use_guidance_stage1': {
        'type': bool,
        'required': False,
        'default': True
    },
    'use_guidance_stage2': {
        'type': bool,
        'required': False,
        'default': False
    },
    'both_stages': {
        'type': bool,
        'required': False,
        'default': True
    },
    'device': {
        'type': str,
        'required': False,
        'default': torch.device("cuda")
    },
    'image_prompt': {
        'type': str,
        'required': False,
        'default': None
    }
}
