''' infer.py for runpod worker '''

import os
import predict

import runpod

runpod.serverless.start()

MODEL = predict.Predictor()
MODEL.setup()


def validator():
    '''
    Model input options and their validations.
    '''
    return {
        'prompt': {
            'type': str,
            'required': True
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
    }


def run(job):
    '''
    Run inference on the model.
    Returns output path, width the seed used to generate the image.
    '''
    job_input = job['input']

    job_input['seed'] = job_input.get('seed', int.from_bytes(os.urandom(2), "big"))

    img_paths = MODEL.predict(
        prompt=job_input["prompt"],
        width=job_input.get('width', 512),
        height=job_input.get('height', 512),
        init_image=job_input.get('init_image', None),
        mask=job_input.get('mask', None),
        prompt_strength=job_input.get('prompt_strength', 0.8),
        num_outputs=job_input.get('num_outputs', 1),
        num_inference_steps=job_input.get('num_inference_steps', 50),
        guidance_scale=job_input.get('guidance_scale', 7.5),
        scheduler=job_input.get('scheduler', "K-LMS"),
        seed=job_input.get('seed', None)
    )

    job_output = []

    for index, img_path in enumerate(img_paths):
        image_url = runpod.serverless.modules.upload(job['id'], img_path, index)

        job_output.append({
            "image": image_url,
            "seed": job_input['seed'] + index
        })

    return job_output
