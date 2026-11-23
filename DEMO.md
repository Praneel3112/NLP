# Demo script and test checklist

Every input and output below was run on the final code (branch `report-aligned`).

## Before the exam (do this the day before)

- [ ] `cd NLP` then `.\.venv\Scripts\Activate.ps1`
- [ ] `python -m unittest discover -s tests` shows **73 tests, OK**
- [ ] `python server.py`, open http://localhost:8000, status pill says **"All models ready"** (about 10 seconds)
- [ ] If you changed `data/lexicon.csv`, the first start retrains the GRU (about 7 minutes). Do that at home, not in the exam
- [ ] XLM-RoBERTa must be in the Hugging Face cache (`C:\Users\<you>\.cache\huggingface`). It downloads once (about 1 GB). Start the server once with internet, then it works offline. If it is missing, the app still runs and Phase 3 falls back to the Phase 1/2 suggestions
- [ ] Keep the terminal running while you present. `Ctrl+C` stops the server
- [ ] If port 8000 is busy: `$env:PORT=8001; python server.py`

## Suggested demo order (about 5 minutes)

Type each line, press **Analyse message**, then open the tabs.

### 1. Spelling correction

| Type | Result (Phase 3) | What to say |
|---|---|---|
| `namast aap kaise ho` | `namaste aap kaise ho` | A dropped letter is a typing slip, fixed by Phase 1 |
| `kya kr rhe ho bhai` | `kya kar rahe ho bhai` | Shorthand is expanded by Phase 0 first |
| `naaaaam kya hai tumhara` | `naam kya hai tumhara` | Letter stretching is fixed (phonetic match) |
| `namsate good morning aap kaise ho` | `namaste good morning aap kaise ho` | Code-mixed: English words are skipped by language ID |
| `how are you doingggg` | unchanged | The original CIE3 system turned this into "ho amar ho". Language ID fixed it |

### 2. Unseen words are protected (the key point)

| Type | Result | What to say |
|---|---|---|
| `mera naam rahul hai` | unchanged | "rahul" is not in the lexicon; it is kept, not forced onto a lookalike |
| `mera naam Priya hai` | unchanged | Capitalised words are treated as names |
| `kal main school jaunga` | unchanged | An English word and an unseen verb form |
| `mujhe samosa pasand hai` | unchanged | An unseen noun |
| `main Mumbai mein rehta hoon` | unchanged | A city name |
| `mera number 98765 hai` and `kya haal hai?` | unchanged | Digits and punctuation survive |

### 3. Valid spelling variants

| Type | Result | What to say |
|---|---|---|
| `mera dost bahut acha hai` | unchanged | "acha" is a valid alternative spelling of "accha" |
| `aap kese ho` | unchanged | "kese" is a valid variant of "kaise" |
| `pyaari si ladki` | unchanged | Doubled long vowels are alternative spellings |

### 4. Emotion from emoji and slang

| Type | Emotion | Why (shown in the "clues" panel) |
|---|---|---|
| `bht gussa aa raha h 😡😡` | Anger, high intensity | "gussa" + two angry emoji, "bht" intensifies |
| `not happy yaar 😔` | Sadness | "happy" is **negated** (crossed out) |
| `love u maa ❤️` | Love | "love" + heart |
| `hi u r so gr8 yaar` | Joy | slang is expanded first, "gr8" -> "great" |
| `mujhe bahut dukh hua 😢` | Sadness | "dukh" + crying face |
| `mujhe samosa pasand hai` | Joy | "pasand" (like) |
| `kal subah meeting hai` | Neutral | no cue, model confident |
| `yeh kitab kahan rakhi hai` | Neutral, **"Not sure"** | no cue and a weak model guess, so it refuses to guess |
| `😂😂😂` | Joy, nothing to correct | emoji only |

Tip: `http://localhost:8000/?text=bht%20gussa%20aa%20raha%20h&tab=emotion` opens a result directly.

## Be ready for these questions

- **"Why is there no 100%?"** The old 100% was 4 easy sentences with a 13-word lexicon. The honest numbers are in
  `REPORT_UPDATES.md`: on the unseen-word set, 39 of 40 blind sentences come through unchanged.
- **"What does Phase 3 do?"** It chooses between equally plausible candidates using the whole sentence. This is
  rare (5 words in 150 noisy sentences), so its measured contribution is small but positive. Example to show:
  `chaaalll phir milll` has several candidate words for `milll`; XLM-RoBERTa picks `mil`.
- **"Why do Phase 2 and 3 not beat Phase 1 on heavy noise?"** They are deliberately cautious, because eager
  correction damages valid unseen words. Table C of `REPORT_UPDATES.md` is the evidence.
- **"What can it not do?"** See the list below.

## Known limits (say them before you are asked)

- A valid word that is one keyboard slip from a *different* lexicon word can be changed
  (measured examples: `priya` -> `periya`, `holi` -> `holo`).
- A valid alternative spelling that is not in `data/variants.csv` and does not follow the long-vowel rule can be changed.
- Words outside the lexicon are preserved, never fixed. Add them to `data/lexicon.csv` to teach the system.
- Heavily corrupted words (three or more mistakes) are mostly not recovered.
- Sarcasm is not detected: `wah great 🙄` reads as joy.
- The emotion model is small (418 training sentences). On text without any emoji, slang or emotion word it may say "Not sure".

## If something goes wrong on the day

| Symptom | Fix |
|---|---|
| Page does not load | `python server.py` must be running in a terminal; use `http://localhost:8000` (not the file) |
| Button stays disabled | The models are still loading. Wait for the green "All models ready" pill |
| Pill says "Backend offline" | The server stopped; restart it |
| Pill says "First run: training Bi-GRU" | `models/gru_25ep.pt` is missing or the lexicon changed; wait about 7 minutes, or restore the file from git |
| Page looks old | Rebuild the UI: `cd web; npm install; npm run build`, then refresh |
| No internet and Phase 3 note says "XLM-RoBERTa unavailable" | Expected if the model was never downloaded; correction still works |
