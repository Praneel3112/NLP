# Transliteration error correction

This repository contains a Python NLP pipeline for Romanized Indic text:
language identification, a slang normalizer (Phase 0), lexicon/edit-distance
correction with phonetic matching, a 3-layer Bi-GRU with attention, XLM-RoBERTa
context ranking, hybrid emotion
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

The correction lexicon is `data/lexicon.csv`: 1,496 common Romanized Hindi,
Kannada, Tamil and Bengali words with their language and meaning (the first 13
are the original CIE3 sample words). `data/variants.csv` lists accepted alternative
spellings (accha / achha / acha); doubled long vowels (paani / pani, saree / sari)
are also treated as spellings of the same word. Variants are never "corrected".

**Safety for unseen words.** A small lexicon cannot contain every name or noun, so
the pipeline is built not to damage valid words it does not know:

- A word is only changed when the difference looks like a *typing slip* (dropped or
  repeated letter, neighbouring key, swapped letters; `src/detection/noise_distance.py`).
  A different word that happens to be one edit away is left alone.
- The clearest slips are applied directly; other candidates need the GRU to confirm
  them. Numbers, symbols, English words, listed variants and capitalised words in the
  middle of a sentence (names) are never touched.
- Words never become targets if they are shorter than three letters.

Phases:

- **Phase 1** matches words to the lexicon by edit distance (at most 2) and by sound
  (a grapheme-to-phoneme code), which also rescues heavily stretched words.
- **Phase 2** is a 3-layer bidirectional GRU encoder with a one-layer GRU decoder and
  Bahdanau attention (64-d embeddings, 256-d hidden states). It is trained for 25
  epochs, batch size 64, Adam with learning rate 0.001, on 5,985 noisy-to-clean pairs
  generated from the lexicon, with a fixed seed. Training takes about seven minutes
  on one CPU thread, so the model is saved in `models/` and reloaded in seconds; its
  checkpoint stores a fingerprint of the training data and settings, so editing the
  lexicon or the noise engine retrains it automatically (add words to
  `data/lexicon.csv` and restart the server).
- **Phase 3** uses the pretrained `xlm-roberta-base` model (not fine-tuned). For each
  word with several plausible candidates it scores them by how well they fit the rest
  of the sentence, together with how plausible the typing slip is. The first start
  downloads the model (about 1 GB, cached afterwards in your Hugging Face cache); if it
  cannot be loaded the app still runs and Phase 3 falls back to the Phase 1/2 suggestions.

`python main.py` prints the metrics for all phases (CER, WER, BLEU, exact match,
precision, recall, F1, semantic similarity, latency) on the demo sentences and on
held-out sets: a noise benchmark (`data/eval_sentences.csv`), unseen-word sentences
(`data/unseen_sentences.csv`, `data/validation_sentences.csv`) and mixed sentences.
`REPORT_UPDATES.md` lists the numbers and the report statements that depend on them;
`DEMO.md` is a verified demo script and test checklist.

Page links: `http://localhost:8000/?text=bht%20gussa%20aa%20raha%20h&tab=emotion`
pre-fills the box, analyses it and opens the Emotion, Slang or Correction tab.

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
- `src/correction/model_store.py` trains the Phase 2 GRU once and reloads it from `models/`.
- `src/correction/context_ranker.py` is Phase 3 (XLM-RoBERTa candidate ranking).
- `src/correction/pipeline.py` loads `data/lexicon.csv` and builds the deterministic
  training-pair generation between the CLI and web backend.

The assignment brief was not present in the workspace; the interface follows
the workflow and language scope described by the existing Python code and
`configs/model_config.yaml`.
