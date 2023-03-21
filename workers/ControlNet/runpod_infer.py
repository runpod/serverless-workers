'''
RunPod | ControlNet | Infer
'''

import os
import base64
import argparse
from io import BytesIO
from subprocess import call

from PIL import Image
import numpy as np

import runpod
from runpod.serverless.utils import rp_download, rp_upload
from runpod.serverless.utils.rp_validator import validate

from cldm.model import create_model, load_state_dict
from ldm.models.diffusion.ddim import DDIMSampler
from utils import get_state_dict_path, download_model, model_dl_urls, annotator_dl_urls

# ---------------------------------------------------------------------------- #
#                                    Schemas                                   #
# ---------------------------------------------------------------------------- #
BASE_SCHEMA = {
    'image_url': {'type': str, 'required': False, 'default': None},
    'image_base64': {'type': str, 'required': False, 'default': None},
    'prompt': {'type': str, 'required': False, 'default': None},
    'a_prompt': {'type': str, 'required': False, 'default': "best quality, extremely detailed"},
    'n_prompt': {'type': str, 'required': False, 'default': "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"},
    'num_samples': {'type': int, 'required': False, 'default': 1, 'constraints': lambda samples: samples in [1, 4]},
    'image_resolution': {'type': int, 'required': False, 'default': 512, 'constraints': lambda resolution: resolution in [256, 512, 768]},
    'ddim_steps': {'type': int, 'required': False, 'default': 20},
    'scale': {'type': float, 'required': False, 'default': 9.0, 'constraints': lambda scale: 0.1 < scale < 30.0},
    'seed': {'type': int, 'required': True},
    'eta': {'type': float, 'required': False, 'default': 0.0},
    'low_threshold': {'type': int, 'required': False, 'default': 100, 'constraints': lambda threshold: 1 < threshold < 255},
    'high_threshold': {'type': int, 'required': False, 'default': 200, 'constraints': lambda threshold: 1 < threshold < 255},
}

CANNY_SCHEMA = BASE_SCHEMA.copy()

DEPTH_SCHEMA = BASE_SCHEMA.copy()
DEPTH_SCHEMA['detect_resolution'] = {'type': int, 'required': False,
                                     'default': 256, 'constraints': lambda resolution: resolution in [128, 256, 384]}

HED_SCHEMA = BASE_SCHEMA.copy()
HED_SCHEMA['detect_resolution'] = {'type': int, 'required': False,
                                   'default': 256, 'constraints': lambda resolution: resolution in [128, 256, 384]}

NORMAL_SCHEMA = BASE_SCHEMA.copy()
NORMAL_SCHEMA['detect_resolution'] = {'type': int, 'required': False,
                                      'default': 256, 'constraints': lambda resolution: resolution in [128, 256, 384]}
NORMAL_SCHEMA['bg_threshold'] = {'type': float, 'required': False,
                                 'default': 0.0, 'constraints': lambda threshold: 0 <= threshold <= 1}

MLSD_SCHEMA = BASE_SCHEMA.copy()
MLSD_SCHEMA['value_threshold'] = {'type': float, 'required': False,
                                  'default': 0.0, 'constraints': lambda threshold: 0 <= threshold <= 1}
MLSD_SCHEMA['detect_resolution'] = {'type': int, 'required': False,
                                    'default': 256, 'constraints': lambda resolution: resolution in [128, 256, 384]}
MLSD_SCHEMA['distance_threshold'] = {'type': float, 'required': False,
                                     'default': 0.1, 'constraints': lambda threshold: 0.01 <= threshold <= 20.0}

SCRIBBLE_SCHEMA = BASE_SCHEMA.copy()

SEG_SCHEMA = BASE_SCHEMA.copy()
SEG_SCHEMA['detect_resolution'] = {'type': int, 'required': False,
                                   'default': 256, 'constraints': lambda resolution: resolution in [128, 256, 384]}

OPENPOSE_SCHEMA = BASE_SCHEMA.copy()
OPENPOSE_SCHEMA['detect_resolution'] = {'type': int, 'required': False,
                                        'default': 256, 'constraints': lambda resolution: resolution in [128, 256, 384]}


def get_image(image_url, image_base64):
    '''
    Get the image from the provided URL or base64 string.
    Returns a PIL image.
    '''
    if image_url is not None:
        image = rp_download.file(image_url)
        image = image['file_path']

    if image_base64 is not None:
        image_bytes = base64.b64decode(image_base64)
        image = BytesIO(image_bytes)

    input_image = Image.open(image)
    input_image = np.array(input_image)

    return input_image


def predict(job):
    '''
    Run a single prediction on the model.
    '''
    job_input = job['input']

    if job_input.get('seed', None) is None:
        job_input['seed'] = np.random.randint(1000000)

    # Check for provided image
    if job_input.get('image_url', None) is None and job_input.get('image_base64', None) is None:
        return {'error': 'No image provided. Please provide an image_url or image_base64.'}
    elif job_input.get('image_url', None) is not None and job_input.get('image_base64', None) is not None:
        return {'error': 'Both image_url and image_base64 provided. Please provide only one.'}

    # ---------------------------------------------------------------------------- #
    #                                     Nets                                     #
    # ---------------------------------------------------------------------------- #

    # ----------------------------------- Canny ---------------------------------- #
    if MODEL_TYPE == "canny":
        canny_validate = validate(job_input, CANNY_SCHEMA)
        if 'errors' in canny_validate:
            return {'error': canny_validate['errors']}
        validated_input = canny_validate['validated_input']

        outputs = process_canny(
            get_image(validated_input['image_url'], validated_input['image_base64']),
            validated_input['prompt'],
            validated_input['a_prompt'],
            validated_input['n_prompt'],
            validated_input['num_samples'],
            validated_input['image_resolution'],
            validated_input['ddim_steps'],
            validated_input['scale'],
            validated_input['seed'],
            validated_input['eta'],
            validated_input['low_threshold'],
            validated_input['high_threshold'],
            model,
            ddim_sampler,
        )

    # ----------------------------------- Depth ---------------------------------- #
    elif MODEL_TYPE == "depth":
        depth_validate = validate(job_input, DEPTH_SCHEMA)
        if 'errors' in depth_validate:
            return {'error': depth_validate['errors']}
        validated_input = depth_validate['validated_input']

        outputs = process_depth(
            get_image(validated_input['image_url'], validated_input['image_base64']),
            validated_input['prompt'],
            validated_input['a_prompt'],
            validated_input['n_prompt'],
            validated_input['num_samples'],
            validated_input['image_resolution'],
            validated_input['detect_resolution'],
            validated_input['ddim_steps'],
            validated_input['scale'],
            validated_input['seed'],
            validated_input['eta'],
            model,
            ddim_sampler,
        )

    # ------------------------------------ Hed ----------------------------------- #
    elif MODEL_TYPE == "hed":
        hed_validate = validate(job_input, HED_SCHEMA)
        if 'errors' in hed_validate:
            return {'error': hed_validate['errors']}
        validated_input = hed_validate['validated_input']

        outputs = process_hed(
            get_image(validated_input['image_url'], validated_input['image_base64']),
            validated_input['prompt'],
            validated_input['a_prompt'],
            validated_input['n_prompt'],
            validated_input['num_samples'],
            validated_input['image_resolution'],
            validated_input['detect_resolution'],
            validated_input['ddim_steps'],
            validated_input['scale'],
            validated_input['seed'],
            validated_input['eta'],
            model,
            ddim_sampler,
        )

    # ---------------------------------- Normal ---------------------------------- #
    elif MODEL_TYPE == "normal":
        normal_validate = validate(job_input, NORMAL_SCHEMA)
        if 'errors' in normal_validate:
            return {'error': normal_validate['errors']}
        validated_input = normal_validate['validated_input']

        outputs = process_normal(
            get_image(validated_input['image_url'], validated_input['image_base64']),
            validated_input['prompt'],
            validated_input['a_prompt'],
            validated_input['n_prompt'],
            validated_input['num_samples'],
            validated_input['image_resolution'],
            validated_input['detect_resolution'],
            validated_input['ddim_steps'],
            validated_input['scale'],
            validated_input['seed'],
            validated_input['eta'],
            validated_input['bg_threshold'],
            model,
            ddim_sampler,
        )

    # ----------------------------------- MLSD ----------------------------------- #
    elif MODEL_TYPE == "mlsd":
        mlsd_validate = validate(job_input, MLSD_SCHEMA)
        if 'errors' in mlsd_validate:
            return {'error': mlsd_validate['errors']}
        validated_input = mlsd_validate['validated_input']

        outputs = process_mlsd(
            get_image(validated_input['image_url'], validated_input['image_base64']),
            validated_input['prompt'],
            validated_input['a_prompt'],
            validated_input['n_prompt'],
            validated_input['num_samples'],
            validated_input['image_resolution'],
            validated_input['detect_resolution'],
            validated_input['ddim_steps'],
            validated_input['scale'],
            validated_input['seed'],
            validated_input['eta'],
            validated_input['value_threshold'],
            validated_input['distance_threshold'],
            model,
            ddim_sampler,
        )

    # --------------------------------- Scribble --------------------------------- #
    elif MODEL_TYPE == "scribble":
        scribble_validate = validate(job_input, SCRIBBLE_SCHEMA)
        if 'errors' in scribble_validate:
            return {'error': scribble_validate['errors']}
        validated_input = scribble_validate['validated_input']

        outputs = process_scribble(
            get_image(validated_input['image_url'], validated_input['image_base64']),
            validated_input['prompt'],
            validated_input['a_prompt'],
            validated_input['n_prompt'],
            validated_input['num_samples'],
            validated_input['image_resolution'],
            validated_input['ddim_steps'],
            validated_input['scale'],
            validated_input['seed'],
            validated_input['eta'],
            model,
            ddim_sampler,
        )

    # ------------------------------------ Seg ----------------------------------- #
    elif MODEL_TYPE == "seg":
        seg_validate = validate(job_input, SEG_SCHEMA)
        if 'errors' in seg_validate:
            return {'error': seg_validate['errors']}
        validated_input = seg_validate['validated_input']

        outputs = process_seg(
            get_image(validated_input['image_url'], validated_input['image_base64']),
            validated_input['prompt'],
            validated_input['a_prompt'],
            validated_input['n_prompt'],
            validated_input['num_samples'],
            validated_input['image_resolution'],
            validated_input['detect_resolution'],
            validated_input['ddim_steps'],
            validated_input['scale'],
            validated_input['seed'],
            validated_input['eta'],
            model,
            ddim_sampler,
        )

    # --------------------------------- Openpose --------------------------------- #
    elif MODEL_TYPE == "openpose":
        openpose_validate = validate(job_input, OPENPOSE_SCHEMA)
        if 'errors' in openpose_validate:
            return {'error': openpose_validate['errors']}
        validated_input = openpose_validate['validated_input']

        outputs = process_pose(
            get_image(validated_input['image_url'], validated_input['image_base64']),
            validated_input['prompt'],
            validated_input['a_prompt'],
            validated_input['n_prompt'],
            validated_input['num_samples'],
            validated_input['image_resolution'],
            validated_input['detect_resolution'],
            validated_input['ddim_steps'],
            validated_input['scale'],
            validated_input['seed'],
            validated_input['eta'],
            model,
            ddim_sampler,
        )

    # outputs from list to PIL
    outputs = [Image.fromarray(output) for output in outputs]

    # save outputs to file
    os.makedirs("tmp", exist_ok=True)
    outputs = [output.save(f"tmp/output_{i}.png") for i, output in enumerate(outputs)]

    for index, output in enumerate(outputs):
        outputs = rp_upload.upload_image(job['id'], f"tmp/output_{index}.png")

    # return paths to output files
    return outputs


# ---------------------------------------------------------------------------- #
#                                     Main                                     #
# ---------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model_type", type=str,
                    default=None, help="Model URL")


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    MODEL_TYPE = args.model_type

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

    runpod.serverless.start({"handler": predict})
