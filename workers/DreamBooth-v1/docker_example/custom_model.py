'''
RunPod | DreamBooth | Custom Model Fetcher
'''

import os
import wget

from subprocess import call


def downloadmodel_hf(Path_to_HuggingFace, huggingface_token=None):
    '''
    Download model from HuggingFace.
    '''
    if huggingface_token:
        authe = f'https://USER:{huggingface_token}@'
    else:
        authe = "https://"

    os.makedirs('/src/stable-diffusion-custom', exist_ok=True)

    print(f"Current working directory: {os.getcwd()}")

    os.chdir("/src/stable-diffusion-custom")
    call("git init", shell=True)
    call("git lfs install --system --skip-repo", shell=True)
    call('git remote add -f origin '+authe+'huggingface.co/'+Path_to_HuggingFace, shell=True)
    call("git config core.sparsecheckout true", shell=True)
    call('echo -e "\nscheduler\ntext_encoder\ntokenizer\nunet\nvae\nmodel_index.json\n!*.safetensors" > .git/info/sparse-checkout', shell=True)
    call("git pull origin main", shell=True)

    print("Successfully downloaded model from HuggingFace.")

    if os.path.exists('unet/diffusion_pytorch_model.bin'):
        call("rm -r .git", shell=True)
        call("rm model_index.json", shell=True)
        wget.download(
            'https://raw.githubusercontent.com/TheLastBen/fast-stable-diffusion/main/Dreambooth/model_index.json')
        os.chdir('/src')

    while not os.path.exists('/workspace/stable-diffusion-custom/unet/diffusion_pytorch_model.bin'):
        os.chdir('/src')

    print("Downloaded model is compatible with DreamBooth.")


def downloadmodel_lnk(CKPT_Link):
    '''
    Download a model from a ckpt link.
    '''
    call("gdown --fuzzy " + CKPT_Link + " -O model.ckpt", shell=True)

    if os.path.exists('model.ckpt'):
        if os.path.getsize("model.ckpt") > 1810671599:
            wget.download(
                'https://github.com/TheLastBen/fast-stable-diffusion/raw/main/Dreambooth/refmdlz')
            call('unzip -o -q refmdlz', shell=True)
            call('rm -f refmdlz', shell=True)
            wget.download(
                'https://raw.githubusercontent.com/TheLastBen/fast-stable-diffusion/main/Dreambooth/convertodiffv1.py')

            call('python convertodiffv1.py model.ckpt stable-diffusion-custom --v1', shell=True)
            call('rm convertodiffv1.py', shell=True)
            call('rm -r refmdl', shell=True)


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

    call("sed -i 's@\"sample_size\": 256,@\"sample_size\": 512,@g' " +
         MODEL_NAME+"/vae/config.json", shell=True)
    return MODEL_NAME
