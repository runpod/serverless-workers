'''
RunPod | Endpoint | EveryDream | runpod_infer.py

This is the inference start for EveryDream.
It will take the input from the API call, and prepare it for the model.
It will then call the model, and return the results.
'''

import os
import re
import subprocess
from importlib import reload

from PIL import Image
from munch import DefaultMunch
from clip_interrogator import Config, Interrogator

import train
import inference

import runpod
from runpod.serverless.utils import rp_download, rp_cleanup, rp_upload
from runpod.serverless.utils.rp_validator import validate


# ---------------------------------------------------------------------------- #
#                                    Schemas                                   #
# ---------------------------------------------------------------------------- #
CONCEPT_SCHEMA = {
    'token_name': {
        'type': str,
        'required': True
    },
    'alias': {
        'type': str,
        'required': True,
        'constraints': lambda concept_description: concept_description in ["man", "woman", "boy", "girl", "cat", "dog", "person", "guy"]
    },
    'autocaption': {
        'type': bool,
        'required': False,
        'default': False
    }
}

TRAIN_SCHEMA = {
    'amp': {
        'type': bool,
        'required': False,
        'default': True
    },
    'batch_size': {
        'type': int,
        'required': False,
        'default': 2
    },
    'ckpt_every_n_minutes': {
        'type': int,
        'required': False,
        'default': 20
    },
    'clip_grad_norm': {
        'type': float,
        'required': False,
        'default': None
    },
    'clip_skip': {
        'type': int,
        'required': False,
        'default': 0,
        'constraints': lambda clip_skip: clip_skip in [0, 1, 2, 3, 4]
    },
    'cond_dropout': {
        'type': float,
        'required': False,
        'default': 0.04,
        'constraints': lambda cond_dropout: 0 <= cond_dropout <= 1
    },
    'data_url': {
        'type': str,
        'required': True
    },
    'disable_textenc_training': {
        'type': bool,
        'required': False,
        'default': False
    },
    'disable_unet_training': {
        'type': bool,
        'required': False,
        'default': False
    },
    'disable_xformers': {
        'type': bool,
        'required': False,
        'default': False
    },
    'flip_p': {
        'type': float,
        'required': False,
        'default': 0.0,
        'constraints': lambda flip_p: 0 <= flip_p <= 1
    },
    'gradient_checkpointing': {
        'type': bool,
        'required': False,
        'default': False
    },
    'grad_accum': {
        'type': int,
        'required': False,
        'default': 1
    },
    'hf_repo_subfolder': {
        'type': str,
        'required': False,
        'default': None
    },
    'lr': {
        'type': float,
        'required': False,
        'default': None
    },
    'lr_decay_steps': {
        'type': int,
        'required': False,
        'default': None
    },
    'lr_scheduler': {
        'type': str,
        'required': False,
        'default': "constant",
        'constraints': lambda lrs: lrs in ["constant", "linear", "cosine", "polynomial"]
    },
    'lr_warmup_steps': {
        'type': int,
        'required': False,
        'default': None
    },
    'max_epochs': {
        'type': int,
        'required': False,
        'default': 300
    },
    'resolution': {
        'type': int,
        'required': False,
        'default': 512,
        'constraints': lambda resolution: resolution in [256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152]
    },
    'resume_ckpt_url': {
        'type': str,
        'required': False,
        'default': "sd_v1-5_vae.ckpt"
    },
    'sample_prompts': {
        'type': list,
        'required': False,
        'default': []
    },
    'sample_steps': {
        'type': int,
        'required': False,
        'default': 250
    },
    'save_full_precision': {
        'type': bool,
        'required': False,
        'default': False
    },
    'save_optimizer': {
        'type': bool,
        'required': False,
        'default': False
    },
    'scale_lr': {
        'type': bool,
        'required': False,
        'default': False
    },
    'seed': {
        'type': int,
        'required': False,
        'default': 555
    },
    'shuffle_tags': {
        'type': bool,
        'required': False,
        'default': False
    },
    'useadam8bit': {
        'type': bool,
        'required': False,
        'default': True
    },
    'rated_dataset': {
        'type': bool,
        'required': False,
        'default': False
    },
    'rated_dataset_target_dropout_percent': {
        'type': int,
        'required': False,
        'default': 50,
        'constraints': lambda dataset_dropout_percent: 0 <= dataset_dropout_percent <= 100
    }
}

INFERENCE_SCHEMA = {
    'prompt': {
        'type': str,
        'required': True
    },
    'negative_prompt': {
        'type': str,
        'required': False,
        'default': None
    },
    'width': {
        'type': int,
        'required': False,
        'default': 512,
        'constraints': lambda width: width in [128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024]
    },
    'height': {
        'type': int,
        'required': False,
        'default': 512,
        'constraints': lambda height: height in [128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024]
    },
    'num_outputs': {
        'type': int,
        'required': False,
        'default': 1,
        'constraints': lambda num_outputs: 1 <= num_outputs <= 10
    },
    'num_inference_steps': {
        'type': int,
        'required': False,
        'default': 50,
        'constraints': lambda num_inference_steps: 1 <= num_inference_steps <= 500
    },
    'guidance_scale': {
        'type': float,
        'required': False,
        'default': 7.5,
        'constraints': lambda guidance_scale: 0 <= guidance_scale <= 20
    },
    'scheduler': {
        'type': str,
        'required': False,
        'default': 'K-LMS',
        'constraints': lambda scheduler: scheduler in ["DDIM", "DDPM", "DPM-M", "DPM-S",  "EULER-A", "EULER-D", "HEUN", "IPNDM", "KDPM2-A", "KDPM2-D", "PNDM", "K-LMS"]
    },
    'seed': {
        'type': int,
        'required': False,
        'default': int.from_bytes(os.urandom(2), "big")
    },
    'passback': {
        'type': str,
        'required': False,
        'default': None
    }
}

S3_SCHEMA = {
    'accessId': {
        'type': str,
        'required': True
    },
    'accessSecret': {
        'type': str,
        'required': True
    },
    'bucketName': {
        'type': str,
        'required': True
    },
    'endpointUrl': {
        'type': str,
        'required': True
    }
}


# ---------------------------------------------------------------------------- #
#                           Define the Runner Handler                          #
# ---------------------------------------------------------------------------- #
def everydream_runner(job):
    '''
    Takes in raw data from the API call, and prepares it for the model.
    Passes the data to the model to get the results.
    Prepares the resulting output to be returned.
    '''
    reload(train)

    job_input = job['input']

    job_output = {}
    job_output['train'] = {}
    job_output['inference'] = []

    # -------------------------------- Validation -------------------------------- #
    # Validate optional concept input
    if 'concept' in job_input:
        validated_concept_input = validate(job_input['concept'], CONCEPT_SCHEMA)
        if 'errors' in validated_concept_input:
            return {"error": validated_concept_input['errors']}
    concept_input = validated_concept_input['validated_input']

    # Validate the training input
    if 'train' not in job_input:
        return {"error": "No training input provided."}

    validated_train_input = validate(job_input['train'], TRAIN_SCHEMA)
    if 'errors' in validated_train_input:
        return {"error": validated_train_input['errors']}
    train_input = validated_train_input['validated_input']

    # Validate the inference input
    if 's3Config' not in job and 'inference' not in job_input:
        return {"error": "Please provide either an inference input or an S3 config."}
    if 'inference' in job_input:
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

    try:
        # --------------------------------- Downloads -------------------------------- #
        # Convert 'data_url' to 'data_root'
        downloaded_input = rp_download.file(train_input['data_url'])
        if downloaded_input['type'] != "zip":
            job_output = {"error": "data_url must be a zip file"}

        train_input['data_root'] = downloaded_input['extracted_path']

        # Download the resume checkpoint, if provided
        if train_input['resume_ckpt_url'] != "sd_v1-5_vae.ckpt":
            # Check if the URL is from huggingface.co, if so, grab the model repo id.
            if re.match(r"huggingface.co", train_input['resume_ckpt_url']):
                url_parts = train_input['resume_ckpt_url'].split("/")
                train_input['resume_ckpt'] = f"{url_parts[-2]}/{url_parts[-1]}"
            else:
                train_input['resume_ckpt'] = rp_download.file(train_input['resume_ckpt_url'])
        else:
            train_input['resume_ckpt'] = train_input['resume_ckpt_url']

        # ---------------------------- Concept Preparation --------------------------- #
        if 'concept' in job_input:
            concept_images = os.listdir(train_input['data_root'])

            with Interrogator(Config(clip_model_name="ViT-L-14/openai")) as ci:
                for index, image in enumerate(concept_images):
                    file_type = image.split(".")[-1]

                    if concept_input['autocaption']:
                        img = Image.open(os.path.join(
                            train_input['data_root'], image)).convert("RGB")

                        caption = ci.interrogate(img)
                        caption = caption.split(",")[0]

                        caption_tokenized = caption.replace(
                            concept_input['alias'],
                            f"{concept_input['token_name']} {concept_input['alias']}"
                        )

                        os.rename(
                            os.path.join(train_input['data_root'], image),
                            os.path.join(train_input['data_root'],
                                         f"{caption_tokenized}.{file_type}")
                        )
                    else:
                        os.rename(
                            os.path.join(train_input['data_root'], image),
                            os.path.join(
                                train_input['data_root'], f"{concept_input['token_name']}_{index}.{file_type}")
                        )

        # ------------------------------- Format Inputs ------------------------------ #
        # train_input['sample_prompts'] -> sample_prompts.txt
        if train_input['sample_prompts']:
            os.makedirs("sample_prompts.txt", exist_ok=True)
            with open(os.path.join("sample_prompts.txt"), "w", encoding="UTF-8") as sample_file:
                for prompt in train_input['sample_prompts']:
                    sample_file.write(f"{prompt}\n")
        else:
            train_input['sample_prompts'] = "sample_prompts.txt"

        # ------------------------------- Set Defaults ------------------------------- #
        # Set default values for optional parameters
        train_input['project_name'] = job['id']
        train_input['gpuid'] = 0
        train_input['logdir'] = "job_files/logs"
        train_input['log_step'] = 25
        train_input['lowvram'] = False
        train_input['notebook'] = False
        train_input['num_workers'] = 0
        train_input['save_ckpt_dir'] = f"job_files/{job['id']}"
        train_input['save_every_n_epochs'] = None
        train_input['wandb'] = False
        train_input['write_schedule'] = False

        os.makedirs(f"job_files/{job['id']}", exist_ok=True)

        # ------------------------------- Run Training ------------------------------- #
        train_input = DefaultMunch.fromDict(train_input)
        train.main(train_input)

        job_output_files = os.listdir(f"job_files/{job['id']}")
        for file in job_output_files:
            if file.endswith(".ckpt"):
                trained_ckpt_file = file
        trained_ckpt = f"job_files/{job['id']}/{trained_ckpt_file}"

        # ------------------------------- Run Inference ------------------------------ #
        if 'inference' in job_input:
            # Convert .ckpt to Diffusers
            os.makedirs(f"job_files/{job['id']}/converted_diffuser", exist_ok=True)
            subprocess.call([
                "python3", "utils/convert_original_stable_diffusion_to_diffusers.py",
                "--scheduler_type=ddim",
                "--original_config_file=v1-inference.yaml",
                "--image_size=512",
                f"--checkpoint_path={trained_ckpt}",
                "--prediction_type=epsilon",
                "--upcast_attn=False",
                f"--dump_path=job_files/{job['id']}/converted_diffuser"
            ])
            ckpt_path = f"job_files/{job['id']}/converted_diffuser"

            # ckpt_dir = f"{next(os.walk('job_files/logs'))[1][0]}/ckpts"
            # ckpt_name = next(os.walk(f"job_files/logs/{ckpt_dir}"))[1][0]
            # ckpt_path = f"job_files/logs/{ckpt_dir}/{ckpt_name}"

            infer_model = inference.Predictor(ckpt_path)
            infer_model.setup()

            for inference_input in job_input['inference']:
                img_paths = infer_model.predict(
                    prompt=inference_input["prompt"],
                    negative_prompt=inference_input["negative_prompt"],
                    width=inference_input['width'],
                    height=inference_input['height'],
                    num_outputs=inference_input['num_outputs'],
                    num_inference_steps=inference_input['num_inference_steps'],
                    guidance_scale=inference_input['guidance_scale'],
                    scheduler=inference_input['scheduler'],
                    seed=inference_input['seed']
                )

                for index, img_path in enumerate(img_paths):
                    image_url = rp_upload.upload_image(job['id'], img_path)

                    job_output['inference'].append({
                        "image": image_url,
                        "prompt": inference_input["prompt"],
                        "negative_prompt": inference_input["negative_prompt"],
                        "width": inference_input['width'],
                        "height": inference_input['height'],
                        "num_inference_steps": inference_input['num_inference_steps'],
                        "guidance_scale": inference_input['guidance_scale'],
                        "scheduler": inference_input['scheduler'],
                        "seed": inference_input['seed'] + index,
                        "passback": inference_input["passback"]
                    })

        # ------------------------------- Upload Files ------------------------------- #
        if 's3Config' in job:

            # Upload the sample images
            if os.path.exists(f"job_files/{job['id']}/logs/samples"):
                sample_images = os.listdir(f"job_files/{job['id']}/logs/samples")
                sample_urls = []
                for sample_image in sample_images:
                    if sample_image.endswith(".jpg"):
                        sample_url = rp_upload.file(
                            f"{job['id']}/{sample_image}", f"job_files/{job['id']}/logs/samples/{sample_image}", s3_config)
                        sample_urls.append(sample_url)
                job_output['train']['samples'] = sample_urls

            # Upload the checkpoint file
            ckpt_url = rp_upload.file(f"{job['id']}.ckpt", trained_ckpt, s3_config)
            job_output['train']['checkpoint_url'] = ckpt_url

    finally:

        # --------------------------------- Clean Up --------------------------------- #
        # rp_cleanup.clean(['job_files'])
        pass

    return job_output


# ---------------------------------------------------------------------------- #
#                               Start the Worker                               #
# ---------------------------------------------------------------------------- #
runpod.serverless.start({"handler": everydream_runner})
