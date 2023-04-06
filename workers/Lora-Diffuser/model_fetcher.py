import torch
from diffusers import StableDiffusionPipeline
StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
