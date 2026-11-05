# Transliteration error correction

This repository contains a Python NLP pipeline for Romanized Indic text:
language identification, lexicon/edit-distance correction, character-level GRU
and Transformer correction, and evaluation metrics.

## Run the web app with the correction backend

Install the project's Python dependencies in a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Start the app and backend from the project directory:

```powershell
python server.py
```

Open [http://localhost:8000](http://localhost:8000). The server exposes the
frontend and its `/api/status` and `/api/correct` endpoints on the same origin.
Phase 2 (GRU) and Phase 3 (Transformer) models train once in the background on
startup; the page shows which model is training and when both are ready. The web
backend trains each model for 20 epochs on the 13-word sample lexicon and 53
generated noisy-to-clean pairs from the project demo. It uses one CPU thread to
keep the small training workload responsive. The original CLI experiment in
`main.py` continues to use its 100-epoch settings. The web models are retrained
each time the server starts because the project does not include saved model
checkpoints.

The models run on CPU and need PyTorch, which is installed as a dependency of
`sentence-transformers` in `requirements.txt`. Startup may take a while. If
training fails, the header shows an error state and the server terminal prints
the cause. Phase 2 preserves known lexicon words and validates GRU predictions
against the sample lexicon; close invalid predictions use the nearest lexicon
word, while unfamiliar words remain unchanged. This guard prevents malformed
output but cannot compensate for the limited training data.

The browser sends entered text to the local backend for inference. Requests are
not stored. The API accepts up to 2,000 characters per request.

Run the Phase 2 safeguard tests with:

```powershell
python -m unittest discover -s tests -v
```

## Deployment

GitHub Pages can serve the static frontend, but it cannot run this Python
backend. To use Phase 2 and Phase 3 online, deploy `server.py` and the Python
dependencies on a Python-capable host, then serve the frontend through that
backend (or configure a same-origin reverse proxy for the API). The bundled
server defaults to `127.0.0.1`; set `HOST` and `PORT` as required by your host.
Do not expose an unauthenticated development server publicly without adding
appropriate production protections.

## Project entry points

- `main.py` runs the original command-line experiment.
- `server.py` serves the frontend and model-backed correction API.
- `index.html`, `styles.css`, and `app.js` implement the responsive web UI.
- `src/correction/pipeline.py` shares the sample lexicon and deterministic
  training-pair generation between the CLI and web backend.

The assignment brief was not present in the workspace; the interface follows
the workflow and language scope described by the existing Python code and
`configs/model_config.yaml`.
