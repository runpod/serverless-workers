''' Inference for the DreamBooth model. '''

import os
import zipfile
import predictor

import runpod
from runpod.serverless.utils import download, upload, validator, rp_cleanup

MODEL = predictor.Predictor()
MODEL.setup()


def run(job):
    '''
    Run inference on the model.

    input format:
    {
        "instance_prompt": str,
        "class_prompt": str,
        "instance_data": str,
        "samples":[
            {
                "save_sample_prompt": str,
                "save_sample_negative_prompt": str,
                "guidance_scale": float,
                "save_inference_steps": int,
                "n_save_sample": int,
            },
             {
                "save_sample_prompt": str,
                "save_sample_negative_prompt": str,
                "save_guidance_scale": float,
                "save_infer_steps": int,
                "n_save_sample": int,
            },
        ]
    }

    '''
    job_input = job['input']

    job_input['seed'] = job_input.get('seed', int.from_bytes(os.urandom(2), "big"))

    job_input["instance_data"] = download.download_input_objects(job_input["instance_data"])[0]

    MODEL.samples = job_input.get("samples", None)

    job_results = MODEL.predict(
        instance_prompt=job_input["instance_prompt"],
        class_prompt=job_input["class_prompt"],
        instance_data=job_input["instance_data"],
        class_data=job_input.get("class_data", None),
        num_class_images=job_input.get("num_class_images", 50),
        pad_tokens=job_input.get("pad_tokens", False),
        with_prior_preservation=job_input.get("with_prior_preservation", True),
        prior_loss_weight=job_input.get("prior_loss_weight", 1.0),
        seed=job_input.get("seed", 512),
        resolution=job_input.get("resolution", 512),
        center_crop=job_input.get("center_crop", False),
        train_text_encoder=job_input.get("train_text_encoder", True),
        train_batch_size=job_input.get("train_batch_size", 1),
        sample_batch_size=job_input.get("sample_batch_size", 2),
        num_train_epochs=job_input.get("num_train_epochs", 1),
        max_train_steps=job_input.get("max_train_steps", 2000),
        gradient_accumulation_steps=job_input.get("gradient_accumulation_steps", 1),
        gradient_checkpointing=job_input.get("gradient_checkpointing", False),
        learning_rate=job_input.get("learning_rate", 1e-6),
        scale_lr=job_input.get("scale_lr", False),
        lr_scheduler=job_input.get("lr_scheduler", "constant"),
        lr_warmup_steps=job_input.get("lr_warmup_steps", 0),
        use_8bit_adam=job_input.get("use_8bit_adam", True),
        adam_beta1=job_input.get("adam_beta1", 0.9),
        adam_beta2=job_input.get("adam_beta2", 0.999),
        adam_weight_decay=job_input.get("adam_weight_decay", 1e-2),
        adam_epsilon=job_input.get("adam_epsilon", 1e-8),
        max_grad_norm=job_input.get("max_grad_norm", 1.0),

        # samples=job_input.get("samples", None),
        save_guidance_scale=job_input.get("save_guidance_scale", 7.5),
        save_sample_prompt=job_input.get("save_sample_prompt", None),
        save_sample_negative_prompt=job_input.get("save_sample_negative_prompt", None),
        n_save_sample=job_input.get("n_save_sample", 1),
        save_infer_steps=job_input.get("save_infer_steps", 50),
    )

    os.makedirs("output_objects", exist_ok=True)

    with zipfile.ZipFile(job_results, 'r') as zip_ref:
        zip_ref.extractall("output_objects")

    img_paths = []
    for file in os.listdir("output_objects/samples"):
        img_paths.append(os.path.join("output_objects/samples", file))

    job_output = []

    image_urls = upload.files(job['id'], img_paths)

    for index, image_url in enumerate(image_urls):
        job_output.append({
            "image": image_url,
            "seed": job_input['seed'] + index
        })

    rp_cleanup.clean(['cog_class_data', 'cog_instance_data', 'checkpoints'])

    return job_output


runpod.serverless.start({"handler": run})
