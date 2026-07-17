'''
RunPod | DreamBooth | Custom Model Fetcher
'''

import os
import re
import wget
import subprocess


# Only allow "org/repo"-style Hugging Face identifiers: no shell metacharacters,
# no path traversal, no flag-injection via a leading "-".
HF_REPO_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+$")


def _run(command, error_prefix):
    '''
    Run a command as an argv list (never shell=True) so that untrusted job
    input interpolated into any argument can't be interpreted as additional
    shell commands.
    '''
    result = subprocess.run(command, shell=False, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"{error_prefix}: {' '.join(command)}\nError message: {result.stderr.decode('utf-8')}")
    return result


def downloadmodel_hf(Path_to_HuggingFace, huggingface_token=None):
    '''
    Download model from HuggingFace.
    '''
    if not HF_REPO_ID_RE.match(Path_to_HuggingFace or ""):
        raise ValueError(
            f"Invalid Hugging Face repo id, expected 'org/repo': {Path_to_HuggingFace!r}")

    if huggingface_token:
        auth = f'https://USER:{huggingface_token}@'
    else:
        auth = "https://"

    custom_path = '/src/stable-diffusion-custom'
    os.makedirs(custom_path, exist_ok=True)

    print(f"Current working directory: {os.getcwd()}")

    os.chdir(custom_path)

    _run(["git", "init"], "Error executing command")
    _run(["git", "lfs", "install", "--system", "--skip-repo"], "Error executing command")
    _run(
        ["git", "remote", "add", "-f", "origin", f'{auth}huggingface.co/{Path_to_HuggingFace}'],
        "Error executing command"
    )
    _run(["git", "config", "core.sparsecheckout", "true"], "Error executing command")

    # Write the sparse-checkout config directly instead of shelling out to
    # `echo ... > file`, which requires shell interpretation of redirection.
    os.makedirs(".git/info", exist_ok=True)
    with open(".git/info/sparse-checkout", "w", encoding="utf-8") as sparse_checkout_file:
        sparse_checkout_file.write(
            "\nscheduler\ntext_encoder\ntokenizer\nunet\nvae\nmodel_index.json\n!*.safetensors\n"
        )

    _run(["git", "pull", "origin", "main"], "Error executing command")

    print("Successfully downloaded model from HuggingFace.")

    if os.path.exists('unet/diffusion_pytorch_model.bin'):
        _run(["rm", "-r", ".git"], "Error executing command")
        _run(["rm", "model_index.json"], "Error executing command")
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
    if not re.match(r"^https?://", ckpt_link or ""):
        raise ValueError(f"Invalid checkpoint link, must be an http(s) URL: {ckpt_link!r}")

    _run(
        ["gdown", "--fuzzy", "-O", "model.ckpt", ckpt_link],
        f"Error downloading model from link: {ckpt_link}"
    )

    if os.path.exists('model.ckpt') and os.path.getsize("model.ckpt") > 1810671599:
        wget.download(
            'https://github.com/CompVis/stable-diffusion/raw/main/configs/stable-diffusion/v1-inference.yaml',
            'config.yaml')
        _run(
            [
                "python", "/src/diffusers/scripts/convert_original_stable_diffusion_to_diffusers.py",
                "--checkpoint_path", "/src/model.ckpt",
                "--dump_path", "/src/stable-diffusion-custom",
                "--original_config_file", "config.yaml"
            ],
            "Error converting checkpoint to diffusers format"
        )


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

    # Modify the config.json file. model_name is one of the two hardcoded
    # paths above (never derived from job input), but this is converted to
    # argv form too so the file has no remaining shell=True usage.
    _run(
        ["sed", "-i", 's@"sample_size": 256,@"sample_size": 512,@g', f"{model_name}/vae/config.json"],
        "Error modifying config.json"
    )

    return model_name
