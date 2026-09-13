# VideoPrism Video-Text FastAPI Service

## 1. Setup

```bash
git clone https://github.com/google-deepmind/videoprism.git videoprism_repo
pip install -r requirements.txt
```

Make sure `constants.py` (with `EMOTION_PROMPTS`) from your original notebook
sits alongside `app.py` — it's imported for the default query set. If it's
missing, the service falls back to a small built-in emotion list.

## 2. GPU checklist

- `jax[cuda12]` in requirements.txt installs the CUDA-enabled jaxlib wheel.
  If your host has CUDA 11 instead of 12, swap it for `jax[cuda11_pip]` and
  the matching `-f` index URL (check the JAX install docs for the current
  one — the URL scheme has changed a few times).
- Confirm the GPU is actually visible before trusting the service:
  ```bash
  python -c "import jax; print(jax.devices())"
  ```
  This should list a `gpu` device, not just `cpu`.
- `XLA_PYTHON_CLIENT_PREALLOCATE=false` is set in `app.py` so JAX doesn't
  grab all GPU memory up front — useful if you're sharing the GPU with
  anything else. Remove it if you want JAX's default (faster allocs, but
  claims ~90% of VRAM immediately).

## 3. Run

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
```

Keep `--workers 1`: each extra worker would load a second full copy of the
model onto the same GPU. Scale by running one process per GPU instead.

## 4. Call it

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@/path/to/video.mp4" \
  -F "queries=playing drums,sitting,playing flute,concert" \
  -F "temperature=0.01"
```

Omit `queries` to score against the default emotion-prompt set from
`constants.py`.

Response:
```json
{
  "top_match": "playing drums",
  "top_similarity": 0.8421,
  "ranked": [
    {"query": "playing drums", "similarity": 0.8421},
    {"query": "concert", "similarity": 0.1104},
    ...
  ],
  "device_used": "cuda:0"
}
```

Health check: `GET /health` — confirms the model is loaded and shows what
JAX devices are visible.