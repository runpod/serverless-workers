'''
RunPod | Inference
'''

import os

import runpod
from runpod.serverless.utils import rp_download, rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate

import predict

MODEL = predict.Predictor()
MODEL.setup()

INPUT_SCHEMA = {
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


def run(job):
    '''
    Run inference on the model.
    '''
    job_input = job['input']

    # Input Validation
    validated_input = validate(job_input, INPUT_SCHEMA)

    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    validated_input = validated_input['validated_input']

    if validated_input['image_prompt'] is not None:
        validated_input['image_prompt'] = rp_download.download_input_objects(
            [validated_input.get('image_prompt', None)])

    if validated_input['seed'] is None:
        validated_input['seed'] = int.from_bytes(os.urandom(2), "big")

    video_file = MODEL.predict(
        prompt=validated_input['prompt'],
        seed=validated_input['seed'],
        translate=validated_input['translate'],
        both_stages=validated_input['both_stages'],
        use_guidance=validated_input['use_guidance'],
        image_prompt=validated_input['image_prompt'],
    )

    print(video_file)
