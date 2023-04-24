'''
RunPod | Kandinsky | Schemas
'''

INPUT_SCHEMA = {
    'prompt': {
        'type': str,
        'required': True,
    },
    'negative_prompt': {
        'type': str,
        'required': False,
        'default': None
    },
    'negative_prior_prompt': {
        'type': str,
        'required': False,
        'default': None
    },
    'negative_decoder_prompt': {
        'type': str,
        'required': False,
        'default': None
    },
    'num_steps': {
        'type': int,
        'required': False,
        'default': 100
    },
    'batch_size': {
        'type': int,
        'required': False,
        'default': 1
    },
    'guidance_scale': {
        'type': float,
        'required': False,
        'default': 4
    },
    'h': {
        'type': int,
        'required': False,
        'default': 768
    },
    'w': {
        'type': int,
        'required': False,
        'default': 768
    },
    'sampler': {
        'type': str,
        'required': False,
        'default': 'p_sampler'
    },
    'prior_cf_scale': {
        'type': float,
        'required': False,
        'default': 4
    },
    'prior_steps': {
        'type': str,
        'required': False,
        'default': "5"
    }
}
