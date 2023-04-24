"""kandinsky2_serverless.py for runpod worker"""

import os
from kandinsky2 import get_kandinsky2
import runpod
from runpod.serverless.utils import rp_download, rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate

from rp_schemas import INPUT_SCHEMA

model = get_kandinsky2(
    'cuda',
    task_type='text2img',
    cache_dir='/kandinsky2',
    model_version='2.1',
    use_flash_attention=False
)


def generate_image(job):
    '''
    Generate an image from text using Kandinsky2
    '''
    job_input = job["input"]

    # Prompt is the new name for text, but we keep text for backwards compatibility
    if job_input.get('text', None) is not None:
        job_input['prompt'] = job_input['text']

    # Sets negative_prior & negative_decoder to negative_prompt if only negative_prompt is provided
    if job_input.get('negative_prompt', None) is not None:
        job_input['negative_prior_prompt'] = job_input['negative_prompt']
        job_input['negative_decoder_prompt'] = job_input['negative_prompt']

    # Input validation
    validated_input = validate(job_input, INPUT_SCHEMA)

    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    validated_input = validated_input['validated_input']

    # Run inference on the model and get the generated images
    images = model.generate_text2img(
        validated_input['prompt'],
        negative_prior_prompt=validated_input['negative_prior_prompt'],
        negative_decoder_prompt=validated_input['negative_decoder_prompt'],
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
    os.makedirs(f"/{job['id']}", exist_ok=True)

    image_path = os.path.join(f"/{job['id']}", "0.png")
    images[0].save(image_path)

    # Upload the output image to the S3 bucket
    image_url = rp_upload.upload_image(job['id'], image_path)

    # Cleanup
    rp_cleanup.clean([f"/{job['id']}"])

    return {"image_url": image_url}


runpod.serverless.start({"handler": generate_image})
