# syntax=docker/dockerfile:1
#
# VideoPrism FastAPI service for HF Spaces (GPU).
# The model runs on JAX + GPU; tensorflow-cpu is pulled only for the
# SentencePiece tokenizer's gfile glue in videoprism/{tokenizers,utils}.py.
#
# Base: python:3.13-slim — JAX's `jax-cuda12-plugin` pip wheel bundles its
# own CUDA 12 runtime libraries (nvidia-*-cu12), so no nvidia/cuda base
# image is needed; the host driver is provided by the HF Space GPU runtime.
FROM python:3.13-slim-bookworm

# System libs needed by opencv (libGL/libglib), TF (libgomp), and cv2
# video decoding (ffmpeg). Debian slim ships none of these by default.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pre-cache the 944MB model checkpoint FIRST, using a lightweight
# huggingface_hub-only install, so this layer survives changes to the full
# requirements file below (changing deps won't re-trigger the 944MB download).
# google/videoprism-lvt-base-f16r288 is public — no token needed for build.
RUN pip install --no-cache-dir huggingface_hub
ENV HF_HOME=/root/.cache/huggingface
ENV HF_HUB_DISABLE_PROGRESS_BARS=1
RUN python -c "\
from huggingface_hub import hf_hub_download; \
p = hf_hub_download('google/videoprism-lvt-base-f16r288', \
                    'flax_lvt_base_f16r288_repeated.npz'); \
print('checkpoint cached at', p)"

# Install full Python deps (cacheable layer — app code changes below don't
# re-trigger the multi-GB CUDA/JAX wheel download).
COPY requirements.app.txt .
RUN pip install --no-cache-dir -r requirements.app.txt

# App code + the videoprism package + bundled SentencePiece tokenizer.
COPY . .

# Point the tokenizer at the locally-bundled copy (avoids the gs:// path
# that needs tensorflow-io-gcs-filesystem, which has no py3.13 wheel).
ENV VIDEOPRISM_TOKENIZER_MODEL_PATH=/app/assets/tokenizer/cc_en.32000.sentencepiece.model
ENV VIDEOPRISM_MODEL=videoprism_lvt_public_v1_base

# HF Spaces expects the app on port 7860.
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
