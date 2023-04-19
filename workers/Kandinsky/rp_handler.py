"""kandinsky2_serverless.py for runpod worker"""

import os
from kandinsky2 import get_kandinsky2
import runpod
from runpod.serverless.utils import rp_download, rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate

model = get_kandinsky2(
    'cuda',
    task_type='text2img',
    cache_dir='/kandinsky2',
    model_version='2.1',
    use_flash_attention=False
)

INPUT_SCHEMA = {
    'text': {
        'type': str,
        'required': True
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


def generate_image(job):
    job_input = job["input"]

    # Input validation
    validated_input = validate(job_input, INPUT_SCHEMA)

    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    validated_input = validated_input['validated_input']

    images = model.generate_text2img(
        validated_input['text'],
        num_steps=validated_input['num_steps'],
        batch_size=validated_input['batch_size'],
        guidance_scale=validated_input['guidance_scale'],
        h=validated_input['h'],
        w=validated_input['w'],
        sampler=validated_input['sampler'],
        prior_cf_scale=validated_input['prior_cf_scale'],
        prior_steps=validated_input['prior_steps']
    )

    # Save the generated image to a file
    output_path = os.path.join("/tmp", f"{job['id']}_output.png")
    images[0].save(output_path)

    # Upload the output image to the S3 bucket
    image_url = rp_upload.upload_image(job['id'], output_path)

    # Cleanup
    rp_cleanup.clean(['/tmp'])

    return {"image_url": image_url}


runpod.serverless.start({"handler": generate_image})
