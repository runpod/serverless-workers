''' Inference for the DreamBooth model. '''

import os
import zipfile
import predictor

import runpod
from runpod.serverless.utils import download, upload, rp_cleanup

MODEL = predictor.Predictor()
MODEL.setup()


def run(job):
    '''
    Run inference using the dreambooth model.

    Job input:
    {
        "id": str,
        "input": {
            "num_train_epochs": int,
            "max_train_steps": int,
            "train_batch_size": int,
            "sample_batch_size": int,
            "gradient_accumulation_steps": int,
            "gradient_checkpointing": bool,
            "learning_rate": float,
            "scale_lr": bool,
            "lr_scheduler": str,
            "lr_warmup_steps": int,
            "resolution": int,
            "center_crop": bool,
            "use_8bit_adam": bool,
            "with_prior_preservation": bool,
            "prior_loss_weight": float,
            "train_text_encoder": bool,
            "pad_tokens": bool,
            "concepts": [
                {
                    "instance_data": str,
                    "class_data": str,
                    "instance_prompt": str,
                    "class_prompt": str,
                    "num_class_images": int,
                    "seed": int,
                }
            ],
            "samples":[
                {
                    "prompt": str,
                    "negative_prompt": str,
                    "guidance_scale": float,
                    "inference_steps": int,
                    "num_outputs": int,
                    "seed": int,
                }
            ]
        }
    }

    '''
    job_input = job['input']

    # Set float inputs
    job_input['learning_rate'] = float(job_input.get("learning_rate", 1e-6))
    job_input['prior_loss_weight'] = float(job_input.get("prior_loss_weight", 1.0))
    job_input['save_guidance_scale'] = float(job_input.get("save_guidance_scale", 7.5))
    job_input['adam_beta1'] = float(job_input.get("adam_beta1", 0.9))
    job_input['adam_beta2'] = float(job_input.get("adam_beta2", 0.999))
    job_input['adam_weight_decay'] = float(job_input.get("adam_weight_decay", 1e-2))
    job_input['adam_epsilon'] = float(job_input.get("adam_epsilon", 1e-8))
    job_input['max_grad_norm'] = float(job_input.get("max_grad_norm", 1.0))

    # Currently only supports one concept
    concept = job_input['concepts'][0]

    # Download input objects
    concept["instance_data"], concept["class_data"] = download.download_input_objects(
        [concept["instance_data"], concept.get("class_data", None)]
    )

    job_input['seed'] = job_input.get('seed', int.from_bytes(os.urandom(2), "big"))

    MODEL.samples = job_input.get("samples", None)

    job_results = MODEL.predict(
        # ---------------------------- Training Parameters --------------------------- #
        # Intervals
        num_train_epochs=job_input.get("num_train_epochs", 1),
        max_train_steps=job_input.get("max_train_steps", 2000),
        # Batching
        train_batch_size=job_input.get("train_batch_size", 1),
        sample_batch_size=job_input.get("sample_batch_size", 2),
        gradient_accumulation_steps=job_input.get("gradient_accumulation_steps", 1),
        gradient_checkpointing=job_input.get("gradient_checkpointing", False),
        # Learning Rate
        learning_rate=job_input['learning_rate'],
        scale_lr=job_input.get("scale_lr", False),
        lr_scheduler=job_input.get("lr_scheduler", "constant"),
        lr_warmup_steps=job_input.get("lr_warmup_steps", 0),
        # Image Processing
        resolution=job_input.get("resolution", 512),
        center_crop=job_input.get("center_crop", False),
        # Tuning
        use_8bit_adam=job_input.get("use_8bit_adam", False),
        with_prior_preservation=concept.get("with_prior_preservation", True),
        prior_loss_weight=job_input['prior_loss_weight'],
        train_text_encoder=job_input.get("train_text_encoder", True),
        pad_tokens=job_input.get("pad_tokens", False),
        # scheduler=job_input.get('scheduler', "DDIM"),

        adam_beta1=job_input['adam_beta1'],
        adam_beta2=job_input['adam_beta2'],
        adam_weight_decay=job_input['adam_weight_decay'],
        adam_epsilon=job_input['adam_epsilon'],
        max_grad_norm=job_input['max_grad_norm'],

        # --------------------------------- Concepts --------------------------------- #
        instance_data=concept["instance_data"],
        class_data=concept.get("class_data", None),
        instance_prompt=concept["instance_prompt"],
        class_prompt=concept["class_prompt"],
        num_class_images=concept.get("num_class_images", 50),
        seed=concept.get("seed", 512),

        # ---------------------------------- Samples --------------------------------- #
        save_guidance_scale=job_input['save_guidance_scale'],
        save_sample_prompt=job_input.get("save_sample_prompt", None),
        save_sample_negative_prompt=job_input.get("save_sample_negative_prompt", None),
        n_save_sample=job_input.get("n_save_sample", 1),
        save_infer_steps=job_input.get("save_infer_steps", 50),
    )

    os.makedirs("output_objects", exist_ok=True)

    with zipfile.ZipFile(job_results['zip'], 'r') as zip_ref:
        zip_ref.extractall("output_objects")

    job_output = {}

    # img_paths = []
    # for file in os.listdir("output_objects/samples"):
    #     img_paths.append(os.path.join("output_objects/samples", file))

    # image_urls = upload.files(job['id'], img_paths)

    # samples_output = []
    # for index, image_url in enumerate(image_urls):
    #     samples_output.append({
    #         "image": image_url,
    #         "seed": job_input['seed'] + index
    #     })

    for index, sample in enumerate(job_results['samples']):
        sample['image'] = upload.upload_image(
            job['id'], f"output_objects/samples/{sample['image']}", index)

    # job_output["samples"] = samples_output
    job_output["samples"] = job_results['samples']

    # Upload trained model weights to user's bucket
    if job.get('s3Config', False):
        print("Uploading model to S3...")
        model_url = upload.bucket_upload(job['id'], [job_results['zip']], job['s3Config'])
        job_output["tuned_model"] = model_url[0]
    else:
        print("No S3 config provided. Skipping model upload.")

    # Cleanup
    rp_cleanup.clean(['cog_class_data', 'cog_instance_data', 'checkpoints', 'input_objects'])

    return job_output


runpod.serverless.start({"handler": run})
