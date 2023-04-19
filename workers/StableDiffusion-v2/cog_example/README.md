# Stable Diffusion v2 Example Using Cog

This is an example of creating a RunPod serverless worker for Stable Diffusion v2 using [cog-stable-diffusion ](https://github.com/replicate/cog-stable-diffusion/tree/38510524cf4f3dc679e5945ebb52feb40d52c1a9) as the base.

## Stable Diffusion v2 Cog model

This is an implementation of the [Diffusers Stable Diffusion v2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    cog run script/download-weights

Then, you can run predictions:

    cog predict -i prompt="monkey scuba diving"

## RunPod Serverless Worker Changes

These are the changes made to the base repo.

- `cog.yaml` - Add `runpod` as a dependency
- Add `runpod_infer.py` file - This defines how the worker interacts with your model, the file name is arbitrary.
