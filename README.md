# Transliteration error correction

This repository contains a Python NLP pipeline for Romanized Indic text:
language identification, a slang normalizer (Phase 0), lexicon/edit-distance
correction, character-level GRU and Transformer correction, hybrid emotion
detection from emoji, slang and code-mixed words, and evaluation metrics.
See `PLAN.md` for a plain-language overview.

## Run the web app with the correction backend

Use Python 3.12 (`py -3.12`). On Windows with Smart App Control enabled, NumPy
builds for very new Python versions can be blocked, and `editdistance` has no
prebuilt wheel for Python 3.13+ (the code then falls back to `rapidfuzz`).

Install the project's Python dependencies in a virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Download NLTK's English word list once. Language identification uses it so that
English words in code-mixed text skip the Indic correction phases (without it, a
143-word fallback list is used and many English words get "corrected"):

```powershell
python -c "import nltk; nltk.download('words')"
```

Build the React + Cult UI frontend once (needs Node.js 20+), and again after
any change in `web/`:

```powershell
cd web
npm install
npm run build
cd ..
```

Start the app and backend from the project directory:

```powershell
python server.py
```

Open [http://localhost:8000](http://localhost:8000). The server serves the built
frontend from `web/dist` (or the original `index.html` page if it has not been
built) and its `/api/status`, `/api/correct` and `/api/emotion/metrics`
endpoints on the same origin. A link such as
`http://localhost:8000/?text=bht%20gussa%20aa%20raha%20h` pre-fills the message
and analyses it once the models are ready.

For frontend development with hot reload, keep `python server.py` running and
run `npm run dev` inside `web/`; Vite proxies `/api` to the Python server.

The emotion classifier (word + character TF-IDF with Logistic Regression,
trained on `data/emotion_dataset.csv`) trains first, in about two seconds.

The correction lexicon is `data/lexicon.csv`: 279 common Romanized Hindi,
Kannada, Tamil and Bengali words with their language and meaning (the first 13
are the original CIE3 sample words). Phase 2 (GRU) and Phase 3 (Transformer) are
trained for 100 epochs on 1,117 noisy-to-clean pairs generated from it, on one
CPU thread. Training takes about 15 minutes, so the trained models are saved in
`models/` and reloaded in seconds on later starts. Each checkpoint stores a
fingerprint of its training data and settings; if you edit the lexicon, the
models retrain automatically on the next start. `main.py` uses the same saved
models.

The models run on CPU and need PyTorch, which is installed as a dependency of
`sentence-transformers` in `requirements.txt`. Startup may take a while. If
training fails, the header shows an error state and the server terminal prints
the cause. Phase 2 preserves known lexicon words and validates GRU predictions
against the sample lexicon; close invalid predictions use the nearest lexicon
word, while unfamiliar words remain unchanged. This guard prevents malformed
output but cannot compensate for the limited training data.

The browser sends entered text to the local backend for inference. Requests are
not stored. The API accepts up to 2,000 characters per request.

Run the tests (correction safeguards, slang normalizer, emotion detection) with:

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
- `web/` holds the React + Tailwind + Cult UI frontend (built to `web/dist`).
- `index.html`, `styles.css`, and `app.js` are the original CIE3 web UI, kept as
  a fallback.
- `src/preprocessing/slang_normalizer.py` is Phase 0 (slang expansion and emoji
  separation).
- `src/emotion/` holds the emotion lexicons, rule scorer, trained classifier and
  hybrid detector.
- `src/correction/model_store.py` trains Phases 2/3 once and reloads them from `models/`.
- `src/correction/pipeline.py` loads `data/lexicon.csv` and builds the deterministic
  training-pair generation between the CLI and web backend.

The assignment brief was not present in the workspace; the interface follows
the workflow and language scope described by the existing Python code and
`configs/model_config.yaml`.
