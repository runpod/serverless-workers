'''
RunPod | Endpoint | Dreambooth

This is the handler for the DreamBooth serverless worker.
'''

import os
import base64
import shutil
import requests
import subprocess
from requests.adapters import HTTPAdapter, Retry

import runpod
from runpod.serverless.utils import rp_download, upload_file_to_bucket, upload_in_memory_object
from runpod.serverless.utils.rp_validator import validate

from rp_dreambooth import dump_only_textenc, train_only_unet
from rp_custom_model import selected_model
from rp_schemas import TRAIN_SCHEMA, INFERENCE_SCHEMA, S3_SCHEMA


automatic_session = requests.Session()
retries = Retry(total=6, backoff_factor=10, status_forcelist=[502, 503, 504])
automatic_session.mount('http://', HTTPAdapter(max_retries=retries))


# ---------------------------------------------------------------------------- #
#                              Automatic Functions                             #
# ---------------------------------------------------------------------------- #
def run_inference(inference_request):
    '''
    Run inference on a request.
    '''
    response = automatic_session.post(url='http://127.0.0.1:3000/sdapi/v1/txt2img',
                                      json=inference_request, timeout=600)
    return response.json()


# ---------------------------------------------------------------------------- #
#                                    Handler                                   #
# ---------------------------------------------------------------------------- #
def handler(job):
    '''
    This is the handler function that will be called on every job.
    '''
    job_input = job['input']

    job_output = {}

    # -------------------------------- Validation -------------------------------- #
    # Validate the training input
    if 'train' not in job_input:
        return {"error": "Missing training input."}
    train_input = job_input['train']

    validated_train_input = validate(job_input['train'], TRAIN_SCHEMA)
    if 'errors' in validated_train_input:
        return {"error": validated_train_input['errors']}
    train_input = validated_train_input['validated_input']

    # Validate the inference input
    if 's3Config' not in job and 'inference' not in job_input:
        return {"error": "Please provide either an inference input or an S3 config."}
    if 'inference' in job_input:

        # Ensure that the inference input is a list
        if not isinstance(job_input['inference'], list):
            return {"error": "Inference input must be a list of inferences."}

        for index, inference_input in enumerate(job_input['inference']):
            validated_inf_input = validate(inference_input, INFERENCE_SCHEMA)
            if 'errors' in validated_inf_input:
                return {"error": validated_inf_input['errors']}
            job_input['inference'][index] = validated_inf_input['validated_input']

    # Validate the S3 config, if provided
    s3_config = None
    if 's3Config' in job:
        validated_s3_config = validate(job['s3Config'], S3_SCHEMA)
        if 'errors' in validated_s3_config:
            return {"error": validated_s3_config['errors']}
        s3_config = validated_s3_config['validated_input']

    # -------------------------- Download Training Data -------------------------- #
    downloaded_input = rp_download.file(train_input['data_url'])

    # Make clean data directory
    allowed_extensions = [".jpg", ".jpeg", ".png"]
    flat_directory = f"job_files/{job['id']}/clean_data"
    os.makedirs(flat_directory, exist_ok=True)

    for root, dirs, files in os.walk(downloaded_input['extracted_path']):
        # Skip __MACOSX folder
        if '__MACOSX' in root:
            continue

        for file in files:
            file_path = os.path.join(root, file)
            if os.path.splitext(file_path)[1].lower() in allowed_extensions:
                shutil.copy(
                    os.path.join(downloaded_input['extracted_path'], file_path),
                    flat_directory
                )

    # Rename the files to the concept name, if provided.
    if train_input['concept_name'] is not None:
        concept_images = os.listdir(flat_directory)
        for index, image in enumerate(concept_images):
            file_type = image.split(".")[-1]
            os.rename(
                os.path.join(flat_directory, image),
                os.path.join(flat_directory,
                             f"{train_input['concept_name']} ({index}).{file_type}")
            )

    os.makedirs(f"job_files/{job['id']}", exist_ok=True)
    os.makedirs(f"job_files/{job['id']}/model", exist_ok=True)

    # ---------------------------- Set Starting Model ---------------------------- #
    if train_input['hf_model'] and train_input['ckpt_link']:
        return {"error": "Please provide either a Hugging Face model or a checkpoint URL."}

    model_name = selected_model(
        train_input['hf_model'],
        train_input['ckpt_link'],
        train_input['hf_token']
    )

    # ----------------------------------- Train ---------------------------------- #
    dump_only_textenc(
        model_name=model_name,
        concept_dir=flat_directory,
        ouput_dir=f"job_files/{job['id']}/model",
        training_steps=train_input['text_steps'],
        PT="",
        seed=train_input['text_seed'],
        batch_size=train_input['text_batch_size'],
        resolution=train_input['text_resolution'],
        precision="fp16",
        learning_rate=train_input['text_learning_rate'],
        lr_scheduler=train_input['text_lr_scheduler'],
        enable_adam=train_input['text_8_bit_adam'],
        pndm_scheduler=train_input['pndm_scheduler']
    )

    train_only_unet(
        stp=500,
        SESSION_DIR="TEST_OUTPUT",
        model_name=model_name,
        INSTANCE_DIR=flat_directory,
        OUTPUT_DIR=f"job_files/{job['id']}/model",
        offset_noise=train_input['offset_noise'],
        PT="",
        seed=train_input['unet_seed'],
        batch_size=train_input['unet_batch_size'],
        resolution=train_input['unet_resolution'],
        precision="fp16",
        num_train_epochs=train_input['unet_epochs'],
        learning_rate=train_input['unet_learning_rate'],
        lr_scheduler=train_input['unet_lr_scheduler'],
        enable_adam=train_input['unet_8_bit_adam'],
        pndm_scheduler=train_input['pndm_scheduler']
    )

    # Convert to CKPT
    diffusers_to_ckpt = subprocess.Popen(
        [
            "python", "/src/diffusers/scripts/convertosdv2.py",
            "--fp16",
            f"/src/job_files/{job['id']}/model",
            f"/src/job_files/{job['id']}/{job['id']}.ckpt"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    stdout, stderr = diffusers_to_ckpt.communicate()
    return_code = diffusers_to_ckpt.returncode

    if return_code != 0:
        print(f"Converting to ckpt stdout: {stdout.decode('utf-8')}")
        return {"error": f"Error converting to CKPT: {stderr.decode('utf-8')}"}

    trained_ckpt = f"/src/job_files/{job['id']}/{job['id']}.ckpt"

    # --------------------------------- Inference -------------------------------- #
    if 'inference' in job_input:
        os.makedirs(f"job_files/{job['id']}/inference_output", exist_ok=True)

        subprocess.Popen([
            "python", "/workspace/sd/stable-diffusion-webui/webui.py",
            "--port", "3000",
            "--nowebui", "--api", "--xformers",
            "--ckpt", f"{trained_ckpt}"
        ])

        # WIP: Upload the object instead of the file
        inference_results = []
        for infer_index, inference in enumerate(job_input['inference']):
            passback = inference.pop('passback')
            results = run_inference(inference)
            results['passback'] = passback

            for index, image in enumerate(results['images']):
                image_bytes = base64.b64decode(image.split(",", 1)[0])
                results['images'][index] = upload_in_memory_object(
                    file_name=f"{infer_index}-{index}.png",
                    file_data=image_bytes,
                    prefix=job['id']
                )

            inference_results.append(results)

        job_output['inference'] = inference_results

    # ------------------------------- Upload Files ------------------------------- #
    if 's3Config' in job:
        job_output['train'] = {}

        # Upload the checkpoint file
        job_output['train']['checkpoint_url'] = upload_file_to_bucket(
            file_name=f"{job['id']}.ckpt",
            file_location=trained_ckpt,
            bucket_creds=s3_config,
            bucket_name=job['s3Config']['bucketName'],
        )

    return job_output


runpod.serverless.start({"handler": handler, "refresh_worker": True})
