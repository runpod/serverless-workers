# RealESRGAN Serverless API for RunPod

Dervied from [kodxana](https://github.com/kodxana/esrgan-container) repository.

This serverless function deploys a RealESRGAN model that can upscale images using various pre-trained models. You can provide a set of parameters to control the upscaling process and choose the output format.

## Required Environment Variables

To deploy this function, you need to provide the following environment variables:

- `BUCKET_ACCESS_KEY_ID`: Your storage Access Key ID for accessing your S3 storage.
- `BUCKET_SECRET_ACCESS_KEY`: Your storage Secret Access Key for accessing your S3 storage.
- `S3_BUCKET_NAME`: The name of the S3 bucket where the results will be stored.
- `BUCKET_ENDPOINT_URL`: The endpoint URL of your S3 storage.

## Input Parameters

The function accepts the following input parameters:

- `data_url` (required, string): The URL of the input image file or a zip file containing multiple image files.
- `model` (optional, string): The pre-trained model to be used for upscaling. Possible values are:
    - `RealESRGAN_x4plus`
    - `RealESRNet_x4plus`
    - `RealESRGAN_x4plus_anime_6B`
    - `RealESRGAN_x2plus`
    Default is `RealESRGAN_x4plus`.
- `scale` (optional, float): The scale factor for upscaling. Must be between 0 and 4. Default is 4.
- `tile` (optional, int): The tile size for the upscaling process. Default is 0 (no tiling).
- `tile_pad` (optional, int): The padding size for the tiles. Default is 10.
- `pre_pad` (optional, int): The padding size before upscaling. Default is 0.
- `output_type` (optional, string): The output format for the results. Possible values are:
    - `individual`: Each image will be stored separately.
    - `zip`: All images will be stored in a zip file.
    Default is `individual`.

## Output

The function returns a list of presigned URLs for the upscaled images. If the `output_type` is set to `individual`, the list contains URLs for each individual image. If the `output_type` is set to `zip`, the list contains a single URL for the zip file containing all the upscaled images.

## Example Usage

To call the function with input parameters:

```json
{
  "input": {
    "data_url": "LINKTOZIP/LINKTOIMAGE",
        "output_type": "zip"
  }
}
```
