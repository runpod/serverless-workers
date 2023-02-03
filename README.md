<div align="center">

<h1>Serverless | Workers</h1>

[![CI | Code Quality](https://github.com/runpod/serverless-workers/actions/workflows/CI_pylint.yml/badge.svg)](https://github.com/runpod/serverless-workers/actions/workflows/CI_pylint.yml)

</div>

Official set of serverless AI workers provided by RunPod.

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Pod Worker Functionality](#pod-worker-functionality)
- [Cog](#cog)
- [Deployed Containers](#deployed-containers)
- [Directory Structure](#directory-structure)

## Pod Worker Functionality

All the workers us the [RunPod Python Package](https://github.com/runpod/runpod-python) to implement the work functions, see the [serverless-worker docs](https://github.com/runpod/runpod-python/blob/main/docs/serverless/worker.md) for more information.

## Cog

To aid in packaging, these workers referenced the [Cog](https://github.com/replicate/cog) package to build the containers. See [cog_setup.md](docs/cog_setup.md) for more information on how to install Cog.

## Deployed Containers

| Model                                      | Docker Hub                                                                                      |
|--------------------------------------------|-------------------------------------------------------------------------------------------------|
| [Stable Diffusion v1](stablediffusion-v1/) | [runpod/ai-api-stable-diffusion-v1](https://hub.docker.com/r/runpod/ai-api-stable-diffusion-v1) |
| [Stable Diffusion v2](StableDiffusion-v2/) | [runpod/ai-api-dreambooth-v1](https://hub.docker.com/r/runpod/ai-api-dreambooth-v1)             |
| [Dream Booth v1](dreambooth-v1/)           | [runpod/ai-api-dream-booth-v1](https://hub.docker.com/r/runpod/ai-api-dream-booth-v1)           |
| [Anything v3](anything-v3/)                | [runpod/ai-api-anything-v3](https://hub.docker.com/r/runpod/ai-api-anything-v3)                 |
| [Openjourney](Openjourney/)                | [runpod/ai-api-openjourney](https://hub.docker.com/r/runpod/ai-api-openjourney)                 |
| [Whisper](Whisper/)                        | [runpod/ai-api-whisper](https://hub.docker.com/r/runpod/ai-api-whisper)                         |
| [helloworld](helloworld/)                  |                                                                                                 |

## Directory Structure

```bash
.
```
