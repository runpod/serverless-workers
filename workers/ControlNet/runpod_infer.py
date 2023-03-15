'''
RunPod | ControlNet | Infer
'''

import os
from subprocess import call

from PIL import Image
import numpy as np

import runpod
from runpod.serverless.utils import rp_download, rp_upload
from runpod.serverless.utils.rp_validator import validate


from cldm.model import create_model, load_state_dict
from ldm.models.diffusion.ddim import DDIMSampler
from utils import get_state_dict_path, download_model, model_dl_urls, annotator_dl_urls


MODEL_TYPE = "openpose"

if MODEL_TYPE == "canny":
    from gradio_canny2image import process_canny
elif MODEL_TYPE == "depth":
    from gradio_depth2image import process_depth
elif MODEL_TYPE == "hed":
    from gradio_hed2image import process_hed
elif MODEL_TYPE == "normal":
    from gradio_normal2image import process_normal
elif MODEL_TYPE == "mlsd":
    from gradio_hough2image import process_mlsd
elif MODEL_TYPE == "scribble":
    from gradio_scribble2image import process_scribble
elif MODEL_TYPE == "seg":
    from gradio_seg2image import process_seg
elif MODEL_TYPE == "openpose":
    from gradio_pose2image import process_pose


model = create_model('./models/cldm_v15.yaml').cuda()
model.load_state_dict(load_state_dict(get_state_dict_path(MODEL_TYPE), location='cuda'))
ddim_sampler = DDIMSampler(model)


def predict(job):
    '''
    Run a single prediction on the model.
    '''
    job_input = job['input']

    num_samples = int(job_input['num_samples'])
    image_resolution = int(job_input['image_resolution'])

    if not job_input['seed']:
        seed = np.random.randint(1000000)
    else:
        seed = int(job_input['seed'])

    image_path = rp_download(job_input['image']['file_path'])

    # load input_image
    input_image = Image.open(image_path)
    # convert to numpy
    input_image = np.array(input_image)

    if MODEL_TYPE == "canny":
        outputs = process_canny(
            input_image,
            job_input['prompt'],
            job_input['a_prompt'],
            job_input['n_prompt'],
            num_samples,
            image_resolution,
            job_input['ddim_steps'],
            job_input['scale'],
            seed,
            job_input['eta'],
            job_input['low_threshold'],
            job_input['high_threshold'],
            model,
            ddim_sampler,
        )
    elif MODEL_TYPE == "depth":
        outputs = process_depth(
            input_image,
            job_input['prompt'],
            job_input['a_prompt'],
            job_input['n_prompt'],
            num_samples,
            image_resolution,
            job_input['detect_resolution'],
            job_input['ddim_steps'],
            job_input['scale'],
            seed,
            job_input['eta'],
            model,
            ddim_sampler,
        )
    elif MODEL_TYPE == "hed":
        outputs = process_hed(
            input_image,
            job_input['prompt'],
            job_input['a_prompt'],
            job_input['n_prompt'],
            num_samples,
            image_resolution,
            job_input['detect_resolution'],
            job_input['ddim_steps'],
            job_input['scale'],
            seed,
            job_input['eta'],
            model,
            ddim_sampler,
        )
    elif MODEL_TYPE == "normal":
        outputs = process_normal(
            input_image,
            job_input['prompt'],
            job_input['a_prompt'],
            job_input['n_prompt'],
            num_samples,
            image_resolution,
            job_input['ddim_steps'],
            job_input['scale'],
            seed,
            job_input['eta'],
            job_input['bg_threshold'],
            model,
            ddim_sampler,
        )
    elif MODEL_TYPE == "mlsd":
        outputs = process_mlsd(
            input_image,
            job_input['prompt'],
            job_input['a_prompt'],
            job_input['n_prompt'],
            num_samples,
            image_resolution,
            job_input['detect_resolution'],
            job_input['ddim_steps'],
            job_input['scale'],
            seed,
            job_input['eta'],
            job_input['value_threshold'],
            job_input['distance_threshold'],
            model,
            ddim_sampler,
        )
    elif MODEL_TYPE == "scribble":
        outputs = process_scribble(
            input_image,
            job_input['prompt'],
            job_input['a_prompt'],
            job_input['n_prompt'],
            num_samples,
            image_resolution,
            job_input['ddim_steps'],
            job_input['scale'],
            seed,
            job_input['eta'],
            model,
            ddim_sampler,
        )
    elif MODEL_TYPE == "seg":
        outputs = process_seg(
            input_image,
            job_input['prompt'],
            job_input['a_prompt'],
            job_input['n_prompt'],
            num_samples,
            image_resolution,
            job_input['detect_resolution'],
            job_input['ddim_steps'],
            job_input['scale'],
            seed,
            job_input['eta'],
            model,
            ddim_sampler,
        )
    elif MODEL_TYPE == "openpose":
        outputs = process_pose(
            input_image,
            job_input['prompt'],
            job_input['a_prompt'],
            job_input['n_prompt'],
            num_samples,
            image_resolution,
            job_input['detect_resolution'],
            job_input['ddim_steps'],
            job_input['scale'],
            seed,
            job_input['eta'],
            model,
            ddim_sampler,
        )

        # outputs from list to PIL
        outputs = [Image.fromarray(output) for output in outputs]

        # save outputs to file
        outputs = [output.save(f"tmp/output_{i}.png") for i, output in enumerate(outputs)]

        for index, output in enumerate(outputs):
            outputs = rp_upload.upload_image(job['id'], f"tmp/output_{index}.png")

        # return paths to output files
        return outputs


runpod.serverless.start({"handler": predict})
