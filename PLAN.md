# Translit Lab v2: Sem-End Plan

This is a plain-language plan for upgrading the CIE3 project for the semester-end exam.

---

## 1. What we have now (CIE3 version)

The project fixes spelling mistakes in **Romanized Indic text**: Hindi, Kannada, Tamil and Bengali written in English letters, like `namast aap kaise ho`.

The text goes through three phases:

| Phase | What it does | How |
|---|---|---|
| **Phase 1** | Finds likely spelling mistakes and fixes the easy ones | Edit distance against a word list (lexicon) |
| **Phase 2** | Fixes words one character at a time | Character-level GRU (an RNN) that we train ourselves |
| **Phase 3** | Fixes the whole sentence | Character-level Transformer, with a lexicon safeguard |

The website is plain HTML, CSS and JavaScript, served by `server.py`.

---

## 2. What we are adding

### ✅ A. Emotion detection from emoji and slang (the teacher's request)
For any message, the app will say **which emotion it shows** and **why**.

Emotions: 😄 joy · 😢 sadness · 😡 anger · 😨 fear · 😲 surprise · ❤️ love · 😐 neutral

Example:
> `bht gussa aa raha h 😡😡`  →  **Anger (91%)**, because 😡 means anger, "gussa" means anger and "bht" (very) makes it stronger.

### ✅ B. Slang Normalizer: a new **Phase 0**
Before correction starts, slang and short forms are turned into normal words:

| You type | Phase 0 turns it into |
|---|---|
| `gr8` | `great` |
| `plz` | `please` |
| `bht` / `bhot` | `bahut` |
| `nhi` / `nai` | `nahi` |
| `h` | `hai` |

Phase 0 also takes emoji out of the text before correction, because a GRU should not try to "spell-correct" 😂. The emoji are kept for emotion detection.

### ✅ C. A new UI built with Cult UI
We rebuild the page in **React + Tailwind + Cult UI components**, with animated, modern components such as a dithering shader hero, animated tabs and animated numbers.

### 🔒 What does NOT change
- **The Phase 1, 2 and 3 correction logic stays exactly the same.** Emotion detection runs **alongside** correction, not inside it.
- The emotion dictionary includes Hindi and other Indic emotion words such as *khush*, *gussa* and *dukhi*. This only helps detect emotions. **It never changes the correction word list, the GRU or the Transformer.**
- If your sentence has no slang and no emoji, Phase 0 passes it through unchanged, so Phases 1 to 3 give **the same output as before**. A test checks this.

---

## 3. How emotion detection works (the "hybrid" approach)

```
           your message: "not happy yaar 😔"
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
  RULE ENGINE (explainable)        ML MODEL (trained by us)
  • emoji dictionary   😔 → sad    • TF-IDF on words (1–2 grams)
  • slang dictionary   lol → joy   • TF-IDF on characters (2–5 grams)
  • Indic words  gussa → anger     • Logistic Regression classifier
  • negation  "not happy" → ✗ joy  • trained on ~420 labelled sentences
  • intensity  CAPS, !!!, sooo,     (English + Hinglish + emoji + slang)
    bahut, ekdum, tumba, romba
        │                                 │
        └──────────► 50% + 50% ◄──────────┘
                         ▼
             FINAL EMOTION + confidence
             + evidence (which words/emoji decided it)
```

**Why two parts?**
- The **rule engine** is easy to explain. It can show exactly which emoji or word gave each emotion. It also handles cases like "not happy".
- The **ML model** learns patterns the rules miss, such as new phrasing, spelling variations and words not in the dictionary. Character n-grams help with spellings like `gusssa` or `khushiii`.
- Together they are **more accurate and still explainable**. If the rules find nothing, the model decides alone.

**How we show that it works:** we test the model on 20% of the dataset that it never saw during training and report **accuracy, macro-F1 and a confusion matrix**.

---

## 4. How the new UI will look

1. **Hero section:** a Cult UI dithering shader background, the title "Translit Lab", a one-line description and badges for the 4 languages.
2. **Live status pill** (Cult UI *Dynamic Island*): shows what the backend is doing, such as *Training emotion model → Training GRU → Training Transformer → Ready*.
3. **Input card:** a text box, character counter, example buttons such as `bht gussa aa raha h 😡😡` and an animated **Analyze** button.
4. **Results:** animated tabs.
   - **Emotion:** a big emoji, the emotion name, an animated confidence %, a bar for every emotion, an intensity badge and evidence chips (😡 → anger, ~~khush~~ negated).
   - **Slang (Phase 0):** your original text next to the cleaned text, with replacements highlighted.
   - **Correction:** Phase 1, 2 and 3 result cards, the final result with a typing animation and a copy button, and a word-by-word explanation.
5. **How it works:** pipeline cards for Phase 0 → 1 → 2 → 3, the emotion branch and the model's accuracy and F1 score.

Cult UI components we plan to use: Hero Dithering, Dynamic Island, Halo Badge, Texture Card, Bg Animate Button, Direction Aware Tabs, Rolling Number, Halo Progress, Text Animate and Copy Button. Expandable Card was dropped: it is built for fixed-size cards, so the word-by-word list uses a simple fold-out section instead.

---

## 5. Folder changes

```
NLP/
├── PLAN.md                          ← this file
├── data/
│   └── emotion_dataset.csv          ← NEW: labelled sentences for training
├── src/
│   ├── preprocessing/
│   │   └── slang_normalizer.py      ← NEW: Phase 0
│   ├── emotion/                     ← NEW package
│   │   ├── lexicons.py              (emoji, slang, Indic emotion words)
│   │   ├── rule_scorer.py           (rule engine + negation + intensity)
│   │   ├── classifier.py            (TF-IDF + Logistic Regression)
│   │   └── detector.py              (combines both)
│   └── correction/service.py        ← adds Phase 0 + emotion to results
├── server.py                        ← serves the new React app
├── web/                             ← NEW: React + Tailwind + Cult UI frontend
└── tests/
    ├── test_slang_normalizer.py     ← NEW
    └── test_emotion.py              ← NEW
```

The old `index.html`, `styles.css` and `app.js` stay as a backup. The server uses them if the new UI has not been built.

---

## 6. How to run (after the upgrade)

```powershell
# 1. Python setup (once) - use Python 3.12
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# 2. Build the new UI (once, or after UI changes)
cd web
npm install
npm run build
cd ..

# 3. Start everything
python server.py
# open http://localhost:8000
```

Run the tests with `python -m unittest discover -s tests -v`.

---

## 7. Order of work

1. Set up a fresh Python environment. The current `venv/` folder points to Python on a different computer, so it does not work here.
2. Phase 0: Slang Normalizer, with tests.
3. Emotion detection: dictionaries → rule engine → dataset → ML model → combined detector, with tests.
4. Connect both to the backend (`service.py`, `server.py`, `main.py`).
5. Build the new React + Cult UI frontend.
6. Update the README and check everything end to end.

---

## 8. Progress (updated 22 Nov 2026)

- ✅ Python environment rebuilt on **Python 3.12**. Python 3.15 was blocked by Windows Smart App Control (NumPy), and `editdistance` has no 3.15 build. The code now falls back to `rapidfuzz` if `editdistance` is missing.
- ✅ Phase 0 slang normalizer
- ✅ Emotion detection: 418 labelled sentences, rule engine, ML model, hybrid
- ✅ Connected to `service.py`, `server.py` and `main.py` (new Step 5)
- ✅ New React + Cult UI frontend in `web/`
- ✅ 20 tests passing (4 original + 16 new)

**Results on 84 unseen test sentences:**

| Approach | Accuracy | Macro-F1 |
|---|---|---|
| Rule engine only | 92.9% | 0.93 |
| ML model only | 71.4% (75.1% with 5-fold cross-validation) | 0.72 |
| **Hybrid (ours)** | **94.0%** | **0.94** |

---

## 9. Viva talking points 🎤

- **Be honest about the rule-engine score:** the same people wrote the dictionaries and the dataset, so rules-only accuracy is probably optimistic. The ML model's 71–75% is the fairer estimate for text unlike our examples. The hybrid takes the best of both.

- **Why hybrid?** Rules are explainable and handle negation. The ML model generalises to new text. Combining them gets the strengths of both.
- **Why TF-IDF + Logistic Regression and not a big Transformer?** Our dataset is small and code-mixed. Pretrained English emotion models don't understand Hinglish. LR trains in about 1 second, works offline and its results are easy to interpret.
- **Why character n-grams?** Romanized Indic text has no fixed spelling (`bahut` / `bahot` / `bhot`). Character n-grams still match these variations.
- **Negation handling:** "khush nahi" and "not happy" should not count as joy. The rules look for a negation word within 2 words of an emotion word.
- **Intensity:** CAPS, `!!!`, stretched letters (`sooo`) and intensifiers (`bahut`, `ekdum`, `tumba`, `romba`) make an emotion stronger.
- **Why Phase 0?** Slang and emoji are noise for spelling correction but signal for emotion. Phase 0 separates them, so each part of the system gets the input it needs.
