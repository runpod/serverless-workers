import runpod
from runpod.serverless.utils import rp_download, rp_cleanup
from runpod.serverless.utils.rp_upload import files, upload_file_to_bucket
from runpod.serverless.utils.rp_validator import validate

import os
import cv2
import zipfile
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from dotenv import load_dotenv
import uuid
import tempfile

load_dotenv()

INPUT_SCHEMA = {
    'data_url': {
        'type': str,
        'required': True
    },
    'model': {
        'type': str,
        'required': False,
        'default': 'RealESRGAN_x4plus',
        'constraints': lambda model: model in [
            'RealESRGAN_x4plus',
            'RealESRNet_x4plus',
            'RealESRGAN_x4plus_anime_6B',
            'RealESRGAN_x2plus',
        ]
    },
    'scale': {
        'type': float,
        'required': False,
        'default': 4,
        'constraints': lambda scale: 0 < scale < 16
    },
    'tile': {
        'type': int,
        'required': False,
        'default': 0,
    },
    'tile_pad': {
        'type': int,
        'required': False,
        'default': 10,
    },
    'pre_pad': {
        'type': int,
        'required': False,
        'default': 0,
    },
    'output_type': {
        'type': str,
        'required': False,
        'default': 'individual',
        'constraints': lambda output_type: output_type in ['individual', 'zip']
    },
}


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"])


def process_image(upsampler, validated_input, image_path, job_id):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    img_mode = 'RGBA' if len(img.shape) == 3 and img.shape[2] == 4 else None

    output, _ = upsampler.enhance(img, outscale=validated_input["scale"])
    imgname, extension = os.path.splitext(os.path.basename(image_path))
    extension = extension[1:]
    max_name_length = 100
    if len(imgname) > max_name_length:
        imgname = str(uuid.uuid4())

    if img_mode == 'RGBA':
        extension = 'png'

    save_path = os.path.join("upscaled", f'{imgname}.{extension}')

    if not os.path.exists("upscaled"):
        os.makedirs("upscaled")

    cv2.imwrite(save_path, output)

    return save_path


def handler(job):
    job_input = job['input']
    validated_input = validate(job_input, INPUT_SCHEMA)
    output_type = validated_input['validated_input'].get('output_type', 'individual')

    if 'errors' in validated_input:
        return {"errors": validated_input['errors']}

    validated_input = validated_input['validated_input']
    remote_file = rp_download.file(validated_input.get('data_url', None))
    data_path = remote_file["file_path"]
    job_id = job['id']

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:

        if validated_input["model"] == 'RealESRGAN_x4plus':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                            num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
        elif validated_input["model"] == 'RealESRNet_x4plus':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                            num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
        elif validated_input["model"] == 'RealESRGAN_x4plus_anime_6B':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                            num_block=6, num_grow_ch=32, scale=4)
            netscale = 4
        elif validated_input["model"] == 'RealESRGAN_x2plus':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                            num_block=23, num_grow_ch=32, scale=2)
            netscale = 2
        else:
            raise "model not found"

    upsampler = RealESRGANer(
        scale=netscale,
        model_path=f'weights/{validated_input["model"]}.pth',
        dni_weight=None,
        model=model,
        tile=validated_input['tile'],
        tile_pad=validated_input['tile_pad'],
        pre_pad=validated_input['pre_pad'],
        half=False,
        gpu_id=None)

    result_paths = []
    if zipfile.is_zipfile(data_path):
        with zipfile.ZipFile(data_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        for file in os.listdir(temp_dir):
            if not file.startswith("__MACOSX") and is_image_file(file):
                result_paths.append(process_image(upsampler, validated_input,
                                    os.path.join(temp_dir, file), job_id))
                os.remove(os.path.join(temp_dir, file))
    else:
        result_paths.append(process_image(upsampler, validated_input, data_path, job_id))

    if output_type == 'zip':
        zip_filename = f"{job_id}_output.zip"
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            for local_path in result_paths:
                zipf.write(local_path)
                os.remove(local_path)
        presigned_url = upload_file_to_bucket(zip_filename, zip_filename)
        os.remove(zip_filename)

        presigned_urls = [presigned_url]
    else:
        presigned_urls = []
        for local_path in result_paths:
            image_urls = files(job['id'], [local_path])
            presigned_urls.extend(image_urls)

    return presigned_urls


runpod.serverless.start({
    "handler": handler
})
