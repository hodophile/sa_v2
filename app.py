"""
FastAPI service wrapping the VideoPrism video-text encoder.

Endpoints
---------
GET  /health   -> basic liveness + device info
POST /predict  -> upload a video, get back ranked similarity against a set
                  of text queries (defaults to the EMOTION_PROMPTS set from
                  the original notebook; override with the `queries` field).

Run with (GPU host):
    uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1

NOTE on --workers: keep this at 1. Each worker process would load its own
copy of the model onto the GPU, and JAX does not share device memory across
processes. If you need more throughput, scale horizontally (one process per
GPU) rather than with multiple workers on the same GPU.
"""

import logging
import os
import sys
import tempfile
import threading
from contextlib import asynccontextmanager
from typing import Optional

# --- Force JAX to see the GPU and to only pre-allocate what it needs. -----
# Must be set before `import jax`.
# os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
import jax.numpy as jnp
import mediapy
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import psutil

# The tokenizer path (vp.load_text_tokenizer) pulls in TensorFlow. Left
# alone, TF initializes its own CUDA context and claims GPU memory
# independently of JAX the moment it's imported/used — the same failure
# mode the original notebook works around with these two lines. Without
# this, TF and JAX silently compete for the same GPU's memory.
# import tensorflow as tf  # noqa: E402

# tf.config.set_visible_devices([], "GPU")
# tf.config.set_visible_devices([], "TPU")

# Make the cloned videoprism repo importable, same as the notebook did.
# sys.path.append("./videoprism_repo")

from videoprism import models as vp  # noqa: E402

# The c4_en SentencePiece model ships at gs://t5-data/... which needs
# tensorflow-io-gcs-filesystem (unavailable on Python 3.13). Bundle a local
# copy and override the path so it works on HF Spaces / any GPU host without
# GCS credentials. Falls back to the gs:// path if the local file is absent
# (e.g. when running inside the original Colab notebook).
_tokenizer_local = os.environ.get("VIDEOPRISM_TOKENIZER_MODEL_PATH", "")
if _tokenizer_local and os.path.exists(_tokenizer_local):
    vp.TEXT_TOKENIZERS["c4_en"]["model_path"] = _tokenizer_local

try:
    from constants import EMOTION_PROMPTS  # noqa: E402
except ImportError:
    EMOTION_PROMPTS = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("videoprism_service")

# ---------------------------------------------------------------------------
# Configuration (env-overridable so you don't have to edit code per deploy)
# ---------------------------------------------------------------------------
MODEL_NAME = os.environ.get("VIDEOPRISM_MODEL", "videoprism_lvt_public_v1_base")
USE_BFLOAT16 = os.environ.get("VIDEOPRISM_BF16", "true").lower() == "true"
NUM_FRAMES = int(os.environ.get("VIDEOPRISM_NUM_FRAMES", "16"))
FRAME_SIZE = int(os.environ.get("VIDEOPRISM_FRAME_SIZE", "288"))
PROMPT_TEMPLATE = os.environ.get("VIDEOPRISM_PROMPT_TEMPLATE", "a video of {}.")
DEFAULT_TEMPERATURE = float(os.environ.get("VIDEOPRISM_TEMPERATURE", "0.01"))
MAX_UPLOAD_MB = int(os.environ.get("VIDEOPRISM_MAX_UPLOAD_MB", "200"))

# A single lock serializes calls into the model. jax.jit'd functions are not
# guaranteed safe to call concurrently from multiple threads against one
# accelerator, and FastAPI runs sync `def` endpoints in a thread pool, so
# without this two simultaneous requests could stomp on each other / OOM
# the GPU.
_inference_lock = threading.Lock()

# Populated at startup.
_state = {
    "flax_model": None,
    "loaded_state": None,
    "text_tokenizer": None,
    "forward_fn": None,
}


def _default_queries() -> list[str]:
    if EMOTION_PROMPTS:
        return [f"{emotion}: {desc}" for emotion, desc in EMOTION_PROMPTS.items()]
    # Fallback if constants.py / EMOTION_PROMPTS isn't available.
    return [
        "happiness",
        "sadness",
        "anger",
        "fear",
        "surprise",
        "disgust",
        "neutral",
    ]


_process = psutil.Process(os.getpid())


def _log_memory(stage: str) -> None:
    rss_gb = _process.memory_info().rss / (1024 ** 3)
    logger.info("MEMORY [%s] | RAM RSS: %.2f GB | JAX devices: %s",
                stage, rss_gb, jax.devices())


def _load_model() -> None:
    logger.info("JAX devices visible: %s", jax.devices())
    gpu_devices = [d for d in jax.devices() if d.platform == "gpu"]
    if not gpu_devices:
        logger.warning(
            "No GPU device visible to JAX. Falling back to whatever "
            "jax.devices() returns (likely CPU). Check your CUDA/jaxlib "
            "install if you expected a GPU here."
        )
    else:
        logger.info("Using GPU device(s): %s", gpu_devices)

    _log_memory("startup")

    fprop_dtype = jnp.bfloat16 if USE_BFLOAT16 else None
    flax_model = vp.get_model(MODEL_NAME, fprop_dtype=fprop_dtype)
    _log_memory("after get_model")

    loaded_state = vp.load_pretrained_weights(MODEL_NAME)
    _log_memory("after load_pretrained_weights (host RAM, pre device_put)")

    # Explicitly commit the weights to the GPU once, here, at load time.
    # Without this, the pretrained weights can sit as plain host-resident
    # arrays that get re-transferred to the device on every single request
    # inside forward_fn — costing both host RAM *and* per-request latency.
    if gpu_devices:
        loaded_state = jax.device_put(loaded_state, gpu_devices[0])
        _log_memory("after device_put to GPU")

    text_tokenizer = vp.load_text_tokenizer("c4_en")

    @jax.jit
    def forward_fn(inputs, text_token_ids, text_paddings, train=False):
        return flax_model.apply(
            loaded_state,
            inputs,
            text_token_ids,
            text_paddings,
            train=train,
        )

    _state["flax_model"] = flax_model
    _state["loaded_state"] = loaded_state
    _state["text_tokenizer"] = text_tokenizer
    _state["forward_fn"] = forward_fn

    # Warm up the jit compile once at startup rather than on the first
    # request, so the first real user isn't the one paying the XLA
    # compilation cost.
    dummy_frames = jnp.zeros((1, NUM_FRAMES, FRAME_SIZE, FRAME_SIZE, 3))
    dummy_queries = _default_queries()
    text_ids, text_paddings = vp.tokenize_texts(text_tokenizer, dummy_queries)
    if USE_BFLOAT16:
        dummy_frames = dummy_frames.astype(jnp.bfloat16)
        text_paddings = text_paddings.astype(jnp.bfloat16)
    forward_fn(dummy_frames, text_ids, text_paddings)
    logger.info("Model loaded and warmed up.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield
    _state.clear()


app = FastAPI(title="VideoPrism Video-Text Service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Video preprocessing (unchanged logic from the notebook)
# ---------------------------------------------------------------------------
def read_and_preprocess_video(
    filename: str, target_num_frames: int, target_frame_size: tuple[int, int]
) -> np.ndarray:
    """Reads only the frames we need, instead of decoding the whole video.

    mediapy.read_video() decodes every frame of the source file into a host
    RAM array before any sampling happens — for a long or high-resolution
    upload that spikes RAM far beyond what the model itself ever needs, and
    is a common cause of an OOM kill that looks like "it's not using the
    GPU". This uses OpenCV to seek directly to the sampled frame indices.
    """
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {filename}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise ValueError("Could not determine frame count for video.")

        frame_indices = np.linspace(
            0, total_frames, num=target_num_frames, endpoint=False, dtype=np.int32
        )

        sampled = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, bgr_frame = cap.read()
            if not ok:
                raise ValueError(f"Failed to read frame {idx} from video.")
            sampled.append(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
        frames = np.array(sampled)
    finally:
        cap.release()

    original_height, original_width = frames.shape[-3:-1]
    target_height, target_width = target_frame_size

    original_aspect_ratio = original_width / original_height
    target_aspect_ratio = target_width / target_height

    if abs(original_aspect_ratio - target_aspect_ratio) > 1e-6:
        if original_aspect_ratio > target_aspect_ratio:
            new_width = int(original_height * target_aspect_ratio)
            offset = (original_width - new_width) // 2
            frames = frames[:, :, offset:offset + new_width, :]
        else:
            new_height = int(original_width / target_aspect_ratio)
            offset = (original_height - new_height) // 2
            frames = frames[:, offset:offset + new_height, :, :]

    frames = mediapy.resize_video(frames, shape=target_frame_size)
    frames = mediapy.to_float01(frames)
    return frames


def compute_similarity_matrix(
    video_embeddings,
    text_embeddings,
    temperature: float,
    apply_softmax: Optional[str] = None,
) -> np.ndarray:
    assert apply_softmax in [None, "over_texts", "over_videos"]
    emb_dim = video_embeddings[0].shape[-1]
    assert emb_dim == text_embeddings[0].shape[-1]

    video_embeddings = np.array(video_embeddings).reshape(-1, emb_dim)
    text_embeddings = np.array(text_embeddings).reshape(-1, emb_dim)
    similarity_matrix = np.dot(video_embeddings, text_embeddings.T)

    if temperature is not None:
        similarity_matrix /= temperature

    if apply_softmax == "over_videos":
        similarity_matrix = np.exp(similarity_matrix)
        similarity_matrix = similarity_matrix / np.sum(
            similarity_matrix, axis=0, keepdims=True
        )
    elif apply_softmax == "over_texts":
        similarity_matrix = np.exp(similarity_matrix)
        similarity_matrix = similarity_matrix / np.sum(
            similarity_matrix, axis=1, keepdims=True
        )

    return similarity_matrix


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------
class RankedResult(BaseModel):
    query: str
    similarity: float


class PredictResponse(BaseModel):
    top_match: str
    top_similarity: float
    ranked: list[RankedResult]
    device_used: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    # HF Spaces readiness probe hits "/"; return 200 so the Space is marked ready.
    return {
        "service": "videoprism-video-text",
        "status": "ok",
        "model_loaded": _state["forward_fn"] is not None,
        "jax_devices": [str(d) for d in jax.devices()],
        "endpoints": ["GET /", "GET /health", "GET /docs", "POST /predict"],
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _state["forward_fn"] is not None,
        "jax_devices": [str(d) for d in jax.devices()],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(
    file: UploadFile = File(..., description="Video file to analyze"),
    queries: Optional[str] = Form(
        None,
        description=(
            "Optional comma-separated text queries to score the video "
            "against. Defaults to the emotion-prompt set if omitted."
        ),
    ),
    temperature: float = Form(DEFAULT_TEMPERATURE),
):
    if _state["forward_fn"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    # Basic size guard so a huge upload doesn't take the process down.
    file.file.seek(0, os.SEEK_END)
    size_mb = file.file.tell() / (1024 * 1024)
    file.file.seek(0)
    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB > {MAX_UPLOAD_MB} MB limit).",
        )

    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(file.file.read())
        tmp.flush()

        _log_memory("before video decode")
        try:
            frames = read_and_preprocess_video(
                tmp.name,
                target_num_frames=NUM_FRAMES,
                target_frame_size=(FRAME_SIZE, FRAME_SIZE),
            )
        except Exception as exc:  # decoding errors, corrupt file, etc.
            raise HTTPException(
                status_code=400, detail=f"Could not read video: {exc}"
            ) from exc
        _log_memory("after video decode")

    text_queries = (
        [q.strip() for q in queries.split(",") if q.strip()]
        if queries
        else _default_queries()
    )
    prompted_queries = [PROMPT_TEMPLATE.format(q) for q in text_queries]

    with _inference_lock:
        text_tokenizer = _state["text_tokenizer"]
        forward_fn = _state["forward_fn"]

        text_ids, text_paddings = vp.tokenize_texts(text_tokenizer, prompted_queries)

        frames_arr = jnp.asarray(frames[None, ...])  # add batch dim
        if USE_BFLOAT16:
            frames_arr = frames_arr.astype(jnp.bfloat16)
            text_paddings = text_paddings.astype(jnp.bfloat16)

        video_embeddings, text_embeddings, _ = forward_fn(
            frames_arr, text_ids, text_paddings
        )

    similarity_matrix = compute_similarity_matrix(
        video_embeddings,
        text_embeddings,
        temperature=temperature,
        apply_softmax="over_texts",
    )

    v2t_similarity_vector = similarity_matrix[0]
    top_indices = np.argsort(v2t_similarity_vector)[::-1]

    ranked = [
        RankedResult(
            query=text_queries[i],
            similarity=float(v2t_similarity_vector[i]),
        )
        for i in top_indices
    ]

    device_used = str(jax.devices()[0]) if jax.devices() else "unknown"

    return PredictResponse(
        top_match=ranked[0].query,
        top_similarity=ranked[0].similarity,
        ranked=ranked,
        device_used=device_used,
    )