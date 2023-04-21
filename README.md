<div align="center">

<h1>Serverless | Endpoint Workers</h1>

[![CI | Code Quality](https://github.com/runpod/serverless-workers/actions/workflows/CI_pylint.yml/badge.svg)](https://github.com/runpod/serverless-workers/actions/workflows/CI_pylint.yml)

</div>

Official set of serverless AI Endpoint workers provided by RunPod. To make requests to the live endpoints please reference our [API docs](https://docs.runpod.io/reference/runpod-apis).

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Pod Worker Functionality](#pod-worker-functionality)
- [Cog](#cog)
- [Deployed Containers](#deployed-containers)
- [Directory Structure](#directory-structure)

## Pod Worker Functionality

All the workers us the [RunPod Python Package](https://github.com/runpod/runpod-python) to implement the work functions.

## Cog

To aid in packaging, these workers referenced the [Cog](https://github.com/replicate/cog) package to build the containers. See [cog_setup.md](docs/cog_setup.md) for more information on how to install Cog.

## Deployed Containers

| Model                                              | Docker Hub                                                                                      |
|----------------------------------------------------|-------------------------------------------------------------------------------------------------|
| [Stable Diffusion v1](workers/StableDiffusion-v1/) | [runpod/ai-api-stable-diffusion-v1](https://hub.docker.com/r/runpod/ai-api-stable-diffusion-v1) |
| [Stable Diffusion v2](workers/StableDiffusion-v2/) | [runpod/ai-api-stable-diffusion-v2](https://hub.docker.com/r/runpod/ai-api-stable-diffusion-v2) |
| [Dream Booth v1](workers/DreamBooth-v1/)           | [runpod/ai-api-dream-booth-v1](https://hub.docker.com/r/runpod/ai-api-dream-booth-v1)           |
| [Anything v3](workers/Anything-v3/)                | [runpod/ai-api-anything-v3](https://hub.docker.com/r/runpod/ai-api-anything-v3)                 |
| [Anything v4](workers/Anything-v4/)                | [runpod/ai-api-anything-v4](https://hub.docker.com/r/runpod/ai-api-anything-v4)                 |
| [Openjourney](workers/Openjourney/)                | [runpod/ai-api-openjourney](https://hub.docker.com/r/runpod/ai-api-openjourney)                 |
| [Whisper](workers/Whisper/)                        | [runpod/ai-api-whisper](https://hub.docker.com/r/runpod/ai-api-whisper)                         |
| [helloworld](helloworld/)                          |                                                                                                 |

## Directory Structure

```bash
.
```
