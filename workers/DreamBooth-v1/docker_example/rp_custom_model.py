'''
RunPod | DreamBooth | Custom Model Fetcher
'''

import os
import wget
import subprocess
from subprocess import call, check_output


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


def downloadmodel_lnk(ckpt_link):
    '''
    Download a model from a ckpt link.
    '''
    result = subprocess.run(
        f"gdown --fuzzy -O model.ckpt {ckpt_link}",
        shell=True, stderr=subprocess.PIPE, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Error downloading model from link: {ckpt_link}\nError message: {result.stderr.decode('utf-8')}")

    if os.path.exists('model.ckpt') and os.path.getsize("model.ckpt") > 1810671599:
        # wget.download('https://github.com/TheLastBen/fast-stable-diffusion/raw/main/Dreambooth/det.py')
        # custom_model_version = check_output(
        #     'python det.py --MODEL_PATH /src/model.ckpt', shell=True).decode('utf-8').replace('\n', '')

        # if custom_model_version == 'v1.5':
        wget.download(
            'https://github.com/CompVis/stable-diffusion/raw/main/configs/stable-diffusion/v1-inference.yaml', 'config.yaml')
        subprocess.run(
            'python /src/diffusers/scripts/convert_original_stable_diffusion_to_diffusers.py --checkpoint_path /src/model.ckpt --dump_path /src/stable-diffusion-custom --original_config_file config.yaml',
            shell=True, check=True)

        # refmdlz_file = 'refmdlz'
        # wget.download(
        #     f'https://github.com/TheLastBen/fast-stable-diffusion/raw/main/Dreambooth/{refmdlz_file}')

        # if not os.path.exists(refmdlz_file):
        #     raise RuntimeError(f"Error downloading {refmdlz_file}")

        # subprocess.run(f'unzip -o -q {refmdlz_file}', shell=True, check=True)
        # subprocess.run(f'rm -f {refmdlz_file}', shell=True, check=True)

        # wget.download(
        #     'https://raw.githubusercontent.com/TheLastBen/fast-stable-diffusion/main/Dreambooth/convertodiffv1.py')

        # # result = subprocess.run(
        # #     'python convertodiffv1.py model.ckpt /src/stable-diffusion-custom --v1',
        # #     shell=True, stderr=subprocess.PIPE, check=False
        # # )
        # result = subprocess.run(
        #     '/src/diffusers/scripts/convert_original_stable_diffusion_to_diffusers.py - -checkpoint_path /src/model.ckpt - -dump_path /src/stable-diffusion-custom - -original_config_file config.yaml ',
        #     shell=True, stderr=subprocess.PIPE, check=False)

        # if result.returncode != 0:
        #     raise RuntimeError(
        #         f"Error executing convert_original_stable_diffusion_to_diffusers.py\nError message: {result.stderr.decode('utf-8')}")

        # subprocess.run('rm convertodiffv1.py', shell=True, check=True)
        # subprocess.run('rm -r refmdl', shell=True, check=True)


def selected_model(path_to_huggingface=None, ckpt_link=None, huggingface_token=None):
    '''
    Either download a model from HuggingFace or from a ckpt link.
    Or use the original V1.5 model.
    '''
    model_name = "/src/stable-diffusion-v1-5"
    os.makedirs("/src/stable-diffusion-custom", exist_ok=True)

    if path_to_huggingface:
        downloadmodel_hf(path_to_huggingface, huggingface_token)
        model_name = "/src/stable-diffusion-custom"
    elif ckpt_link:
        downloadmodel_lnk(ckpt_link)
        model_name = "/src/stable-diffusion-custom"

    # Modify the config.json file
    result = subprocess.run(
        f"sed -i 's@\"sample_size\": 256,@\"sample_size\": 512,@g' {model_name}/vae/config.json",
        shell=True, stderr=subprocess.PIPE, check=False
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Error modifying config.json\nError message: {result.stderr.decode('utf-8')}")

    return model_name
