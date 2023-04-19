''' infer.py for runpod worker '''

import os

from aitemplate.testing.benchmark_pt import benchmark_torch_function
from diffusers import DDIMScheduler
from pipeline_stable_diffusion_ait import StableDiffusionAITPipeline

import runpod
from runpod.serverless.utils import upload, validator
import torch


INPUT_VALIDATIONS = {
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

model_id = "stabilityai/stable-diffusion-2"
scheduler = DDIMScheduler.from_pretrained(
    model_id, subfolder="scheduler", cache_dir="diffusers-cache")

pipe = StableDiffusionAITPipeline.from_pretrained(
    model_id,
    scheduler=scheduler,
    revision="fp16",
    torch_dtype=torch.float16,
    local_files_only=True
).to("cuda")


def run(job):
    '''
    Run inference on the model.
    Returns output path, width the seed used to generate the image.
    '''
    job_input = job['input']

    input_errors = validator.validate(job_input, INPUT_VALIDATIONS)
    if input_errors:
        return {
            "error": input_errors
        }

    job_input['seed'] = job_input.get('seed', int.from_bytes(os.urandom(2), "big"))

    output = pipe(
        prompt=job_input["prompt"],
        negative_prompt=job_input.get("negative_prompt", None),
        width=job_input.get('width', 512),
        height=job_input.get('height', 512),
        num_outputs=job_input.get('num_outputs', 1),
        num_inference_steps=job_input.get('num_inference_steps', 50),
        guidance_scale=job_input.get('guidance_scale', 7.5),
        seed=job_input.get('seed', None)
    )

    img_paths = []
    for i, sample in enumerate(output.images):
        if output.nsfw_content_detected and output.nsfw_content_detected[i] and self.NSFW:
            continue

        output_path = f"/tmp/out-{i}.png"
        sample.save(output_path)
        img_paths.append(output_path)

    job_output = []

    for index, img_path in enumerate(img_paths):
        image_url = upload.upload_image(job['id'], img_path, index)

        job_output.append({
            "image": image_url,
            "seed": job_input['seed'] + index
        })

    return job_output


runpod.serverless.start({"handler": run})
