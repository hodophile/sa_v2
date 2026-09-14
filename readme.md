---
title: Sa Smp 001
emoji: 🎬
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
---

# VideoPrism Video-Text Service

FastAPI service wrapping Google DeepMind's [VideoPrism](https://github.com/google-deepmind/videoprism)
video-text encoder. Upload a video; it returns ranked similarity scores
against a set of text queries (defaults to an emotion-prompt set).

## Endpoints

| Method | Path      | Description |
|--------|-----------|-------------|
| GET    | `/health` | liveness + JAX device info |
| POST   | `/predict` | upload a video, get ranked text-query similarities |

### `/predict` params

- `file` (required): video file
- `queries` (optional): comma-separated text queries
- `temperature` (optional, default `0.01`)

## Running locally

```bash
pip install -r requirements.app.txt
uvicorn app:app --host 0.0.0.0 --port 7860 --workers 1
```

The model (JAX + GPU) and the SentencePiece tokenizer are loaded once at
startup. `tensorflow-cpu` is a dependency only because the videoprism
tokenizer glue uses `tensorflow.io.gfile`; the model compute itself runs
on JAX.
