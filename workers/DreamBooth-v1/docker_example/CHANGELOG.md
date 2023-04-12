# 3/30/23

- Changed Docker base from `runpod/pytorch:3.10-1.13.1-116` to `pytorch:3.10-2.0.0-116-nightly`
- Updated transformers from `4.21.1` to `4.25.1`
- Update TLB diffusers branch from `updt` to `main`
- in drambooth.py changed `train_dreambooth.py` to `train_dreambooth_rnpdendpt.py`
- Added and exposed `pndm_scheduler` in drambooth.py
- Update the mainrunpodA1111.py file to https://huggingface.co/datasets/TheLastBen/RNPD/blob/e27f62ff38f3922365879b5a7543877e0e619a43/Scripts/mainrunpodA1111.py

# 4/10/2023

- Merged PR that corrected denoising_strength type from in to float.
- Removed start.sh and consolidated into Dockerfile
- Update runpod to use main version, fix breaking changes
- Refresh worker after failed jobs
