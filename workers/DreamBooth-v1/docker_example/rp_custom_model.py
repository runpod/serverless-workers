'''
RunPod | DreamBooth | Custom Model Fetcher
'''

import os
import wget
import subprocess
from subprocess import call


def downloadmodel_hf(Path_to_HuggingFace, huggingface_token=None):
    '''
    Download model from HuggingFace.
    '''
    if huggingface_token:
        auth = f'https://USER:{huggingface_token}@'
    else:
        auth = "https://"

    custom_path = '/src/stable-diffusion-custom'
    os.makedirs(custom_path, exist_ok=True)

    print(f"Current working directory: {os.getcwd()}")

    os.chdir(custom_path)
    commands = [
        "git init",
        "git lfs install --system --skip-repo",
        f'git remote add -f origin {auth}huggingface.co/{Path_to_HuggingFace}',
        "git config core.sparsecheckout true",
        'echo -e "\nscheduler\ntext_encoder\ntokenizer\nunet\nvae\nmodel_index.json\n!*.safetensors" > .git/info/sparse-checkout',
        "git pull origin main"
    ]

    for command in commands:
        result = subprocess.run(command, shell=True, stderr=subprocess.PIPE, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"Error executing command: {command}\nError message: {result.stderr.decode('utf-8')}")

    print("Successfully downloaded model from HuggingFace.")

    if os.path.exists('unet/diffusion_pytorch_model.bin'):
        call("rm -r .git", shell=True)
        call("rm model_index.json", shell=True)
        wget.download(
            'https://raw.githubusercontent.com/TheLastBen/fast-stable-diffusion/main/Dreambooth/model_index.json')
        os.chdir('/src')

    while not os.path.exists('/src/stable-diffusion-custom/unet/diffusion_pytorch_model.bin'):
        os.chdir('/src')

    print("Downloaded model is compatible with DreamBooth.")


def downloadmodel_lnk(CKPT_Link):
    '''
    Download a model from a ckpt link.
    '''
    result = subprocess.run(
        f"gdown --fuzzy {CKPT_Link} -O model.ckpt",
        shell=True, stderr=subprocess.PIPE, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Error downloading model from link: {CKPT_Link}\nError message: {result.stderr.decode('utf-8')}")

    if os.path.exists('model.ckpt'):
        if os.path.getsize("model.ckpt") > 1810671599:
            wget.download(
                'https://github.com/TheLastBen/fast-stable-diffusion/raw/main/Dreambooth/refmdlz')
            subprocess.run('unzip -o -q refmdlz', shell=True, check=False)
            subprocess.run('rm -f refmdlz', shell=True, check=False)
            wget.download(
                'https://raw.githubusercontent.com/TheLastBen/fast-stable-diffusion/main/Dreambooth/convertodiffv1.py')

            subprocess.run(
                'python convertodiffv1.py model.ckpt stable-diffusion-custom --v1',
                shell=True, stderr=subprocess.PIPE, check=False
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Error executing convertodiffv1.py\nError message: {result.stderr.decode('utf-8')}")

            subprocess.run('rm convertodiffv1.py', shell=True, check=False)
            subprocess.run('rm -r refmdl', shell=True, check=False)


def selected_model(Path_to_HuggingFace, CKPT_Link, huggingface_token=None):
    '''
    Either download a model from HuggingFace or from a ckpt link.
    Or use the original V1.5 model.
    '''
    MODEL_NAME = "/src/stable-diffusion-v1-5"
    if Path_to_HuggingFace:
        downloadmodel_hf(Path_to_HuggingFace, huggingface_token)
        MODEL_NAME = "/src/stable-diffusion-custom"
    elif CKPT_Link:
        downloadmodel_lnk(CKPT_Link)
        MODEL_NAME = "/src/stable-diffusion-custom"

    result = subprocess.run(
        f"sed -i 's@\"sample_size\": 256,@\"sample_size\": 512,@g' {MODEL_NAME}/vae/config.json",
        shell=True, stderr=subprocess.PIPE, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Error modifying config.json\nError message: {result.stderr.decode('utf-8')}")

    return MODEL_NAME
