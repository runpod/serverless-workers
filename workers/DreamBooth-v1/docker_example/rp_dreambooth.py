'''
Trains DreamBooth image encoder then text encoder sequentially.
'''

import subprocess


# ---------------------------------------------------------------------------- #
#                                 Text Encoder                                 #
# ---------------------------------------------------------------------------- #
def dump_only_textenc(
        model_name, concept_dir, ouput_dir, PT, seed, batch_size, resolution,
        precision, training_steps, learning_rate, lr_scheduler, enable_adam, pndm_scheduler):
    '''
    Train the text encoder first.
    '''
    text_options = [
        "accelerate", "launch", "/src/diffusers/examples/dreambooth/train_dreambooth_rnpdendpt.py",
        "--train_text_encoder",
        "--dump_only_text_encoder",
        f"--pretrained_model_name_or_path={model_name}",
        f"--instance_data_dir={concept_dir}",
        f"--instance_prompt={PT}",
        f"--output_dir={ouput_dir}",
        f"--seed={seed}",
        f"--resolution={resolution}",
        f"--train_batch_size={batch_size}",
        f"--max_train_steps={training_steps}",
        "--gradient_accumulation_steps=1",
        # "--gradient_checkpointing",
        f"--learning_rate={learning_rate}",
        f"--lr_scheduler={lr_scheduler}",
        "--lr_warmup_steps=0",
        f"--mixed_precision={precision}",
        "--image_captions_filename"
    ]

    if enable_adam:
        text_options.append("--use_8bit_adam")

    if pndm_scheduler:
        text_options.append("--PNDM")

    text_encoder = subprocess.Popen(text_options)

    text_encoder.wait()


# ---------------------------------------------------------------------------- #
#                                     UNet                                     #
# ---------------------------------------------------------------------------- #
def train_only_unet(
        stp, SESSION_DIR, model_name, INSTANCE_DIR, OUTPUT_DIR, offset_noise, PT, seed, batch_size,
        resolution, precision, num_train_epochs, learning_rate, lr_scheduler, enable_adam, pndm_scheduler):
    '''
    Train only the image encoder.
    '''
    unet_options = [
        "accelerate", "launch", "/src/diffusers/examples/dreambooth/train_dreambooth_rnpdendpt.py",
        "--image_captions_filename",
        "--train_only_unet",
        f"--save_n_steps={stp}",

        f"--pretrained_model_name_or_path={model_name}",
        f"--instance_data_dir={INSTANCE_DIR}",
        f"--output_dir={OUTPUT_DIR}",
        f"--instance_prompt={PT}",
        f"--seed={seed}",
        f"--resolution={resolution}",
        f"--mixed_precision={precision}",
        f"--train_batch_size={batch_size}",
        "--gradient_accumulation_steps=1",
        f"--learning_rate={learning_rate}",
        f"--lr_scheduler={lr_scheduler}",
        "--lr_warmup_steps=0",

        f"--num_train_epochs={num_train_epochs}",

        f"--Session_dir={SESSION_DIR}"
    ]

    if offset_noise:
        unet_options.append("--offset_noise")

    if enable_adam:
        unet_options.append("--use_8bit_adam")

    if pndm_scheduler:
        unet_options.append("--PNDM")

    unet = subprocess.Popen(unet_options)

    unet.wait()
