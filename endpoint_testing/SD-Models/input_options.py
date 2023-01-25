'''
This file contains the input options for the endpoint testing.
'''

acceptable_inputs = {
    'prompt': {
        'type': str,
        'required': True
    },
    'negative_prompt': {
        'type': str,
        'required': False
    },
    'width': {
        'type': int,
        'required': False
    },
    'height': {
        'type': int,
        'required': False
    },
    'init_image': {
        'type': str,
        'required': False
    },
    'mask': {
        'type': str,
        'required': False
    },
    'prompt_strength': {
        'type': float,
        'required': False
    },
    'num_outputs': {
        'type': int,
        'required': False
    },
    'num_inference_steps': {
        'type': int,
        'required': False
    },
    'guidance_scale': {
        'type': float,
        'required': False
    },
    'scheduler': {
        'type': str,
        'required': False
    },
    'seed': {
        'type': int,
        'required': False
    },
    'nsfw': {
        'type': bool,
        'required': False
    }
}
