
import torch
import runpod
from runpod.serverless.utils import rp_upload

from diffusers import StableDiffusionPipeline


model_path = "sayakpaul/sd-model-finetuned-lora-t4"
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
pipe.unet.load_attn_procs(model_path)
pipe.to("cuda")


def handler(job):
    job_input = job['input']

    image = pipe(
        job_input['prompt'],
        num_inference_steps=job.get('infra_steps', 30),
        guidance_scale=job.get('guidance_Scale', 7.5)
    ).images[0]

    image.save("output.png")

    img_url = rp_upload.upload_image(job['id'], "output.png")

    return {"image_url": img_url}


runpod.serverless.start({"handler": handler})
