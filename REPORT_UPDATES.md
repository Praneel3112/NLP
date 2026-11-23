# Report updates: what the code now does, and what to change in the report

Everything below comes from running `python main.py` on this branch (`report-aligned`).
The numbers are reproducible: the noise seeds, the GRU seed and the saved checkpoint
(`models/gru_25ep.pt`) are fixed, and Phase 1 and the XLM-RoBERTa ranker are deterministic.
Any number you put in the report should be one you can regenerate with that command.

## 1. What now matches the report

| Report says | Code now |
|---|---|
| Noise: vowel deletion (p = 0.25), repeated letters (p = 0.15), phonetic substitutions, plus insertion, substitution, deletion, transposition | All of these. The four typing mistakes hit about 25% of words with 3+ letters (`NoiseGenerator.apply_typos`) |
| LID filter with an English dictionary (NLTK words) | Yes. English words skip Phases 1-3 and stay visible to Phase 3 as context |
| Phase 1: edit distance (tau = 2) + G2P phonetic matching | Yes. Equal-distance ties are broken by phonetic similarity, and heavily distorted words are rescued by matching their phonetic code (e.g. `mmmmina` -> `meena`) |
| Phase 2: 3-layer **bidirectional** GRU encoder, 1-layer GRU decoder, Bahdanau attention | Yes |
| Embedding 64, hidden 256, Adam, lr 0.001, cross-entropy with padding mask, batch 64, 25 epochs | Yes |
| Phase 3: pretrained multilingual transformer for context-aware correction | `xlm-roberta-base`, used as a **candidate ranker** (not fine-tuned) |
| Metrics: CER, WER, BLEU, EM, precision, recall, F1, Scos (all-MiniLM-L6-v2) | All computed by `main.py`. Precision/recall/F1 are token-level (definition in `src/evaluation/metrics.py`) |
| Latency | Measured per phase on this machine's CPU |

## 2. Numbers for the results chapter

### A. The 4 demo sentences (the table the report currently shows)

| System | CER | WER | BLEU | EM % | Prec | Recall | F1 | Scos |
|---|---|---|---|---|---|---|---|---|
| Noisy input | 0.2424 | 0.8462 | 0.0568 | 0.00 | 0.0000 | 0.0000 | 0.0000 | 0.6558 |
| Phase 1: edit distance + G2P | 0.1212 | 0.3846 | 0.2217 | 0.00 | 0.7500 | 0.5455 | 0.6316 | 0.8442 |
| Phase 2: Bi-GRU + attention | 0.1212 | 0.3846 | 0.2217 | 0.00 | 0.7500 | 0.5455 | 0.6316 | 0.8442 |
| Phase 3: XLM-RoBERTa ranker | 0.1667 | 0.4615 | 0.1933 | 0.00 | 0.6250 | 0.4545 | 0.5263 | 0.7959 |

Four sentences are too few to conclude anything. Phase 3 is worse here because on
`ennnn pyar miona` (reference `en peyar meena`) Phase 1 and Phase 2 already hand it the
wrong neighbours, and it then picks `kitna` over `meena`, both equally close to `miona`.

### B. Held-out benchmark (use this one as the main result)

25 clean sentences x 3 fresh noisy versions = **75 test sentences** (from
`data/eval_sentences.csv`, "test" split). The noisy versions use noise seeds
that training never saw. The Phase 3 weights were chosen on the other half ("tune" split).

| System | CER | WER | BLEU | EM % | Prec | Recall | F1 | Scos |
|---|---|---|---|---|---|---|---|---|
| Noisy input | 0.3421 | 0.7636 | 0.0664 | 1.33 | 0.0000 | 0.0000 | 0.0000 | 0.6900 |
| Phase 1: edit distance + G2P | 0.1014 | 0.2481 | 0.3780 | 45.33 | 0.8526 | 0.6751 | 0.7535 | 0.8938 |
| Phase 2: Bi-GRU + attention | 0.1123 | 0.2519 | 0.3721 | 48.00 | 0.8800 | 0.6701 | 0.7608 | 0.8936 |
| Phase 3: XLM-RoBERTa ranker | **0.0904** | **0.2287** | **0.3995** | **50.67** | 0.8625 | **0.7005** | **0.7731** | **0.9077** |

Latency on CPU (one thread), per sentence: Phase 1 1.6 ms, Phase 2 8.4 ms, Phase 3 170.7 ms,
total 180.8 ms.

Honest reading: each phase helps a little more than the one before. The gains from
Phase 2 and Phase 3 are small (a few points of exact match), and the noise is heavy
(about 76% of words are corrupted), so exact match tops out around 50%.

Epoch check (raw GRU word accuracy on noisy held-out words): 25 epochs 72.5%, 50 epochs
75.4%, 100 epochs 74.8%. More training does not help, so 25 epochs is kept.

### C. Emotion detection (new, not in the report)

Hybrid 94.0% accuracy / 0.94 macro-F1 on 84 unseen sentences (rules only 92.9%, model
only 71.4%). The rules score is optimistic because the same people wrote the dictionaries
and the test sentences.

## 3. Report statements that the code cannot support

Change or remove these, or the examiner may ask for evidence you do not have.

| In the report / paper | Problem | Suggested replacement |
|---|---|---|
| Phase 3 "fine-tuned IndicXlit / XLM-RoBERTa", "multilingual seq2seq Transformer" | Phase 3 is XLM-RoBERTa used without fine-tuning, to rank candidates. IndicXlit does Roman to native-script transliteration, a different task | "A pretrained XLM-RoBERTa-base model scores candidate corrections in the context of the sentence" |
| "Fine-tuning optimization AdamW, lr 5e-5, weight decay 0.01", "Batch size 16" for Phase 3 | Nothing is fine-tuned | Remove, or say "no fine-tuning" |
| IndicGLUE / IndicXlit corpus, **500,000** pairs, 80:10:10 split (400k / 50k / 50k) | The project trains on **1,117** pairs generated from a 279-word lexicon and tests on 75 generated sentences | "1,117 synthetic noisy-to-clean word pairs generated from a 279-word lexicon; evaluation on 75 held-out noisy sentences" |
| "150,000-word dictionary", "under 1.8 ms per word" | The lexicon has 279 words. The English list (NLTK, about 234,000 words) is used by language ID | "279-word Romanized Indic lexicon; English filter from the NLTK words corpus" |
| LID "96.2% accuracy / precision" | Never measured | Remove, or measure it on a labelled token set |
| RTX 3090, CUDA 11.8, TensorRT, FP16, Intel i7-12700K, "42.6 ms" latency | The code runs on CPU, and no GPU code exists | Report the measured CPU latency above and your actual hardware |
| "Python 3.10 / PyTorch 2.0 / Ubuntu 22.04" | The project runs on Python 3.12, PyTorch 2.x CPU, Windows 11 | Use your real environment |
| Table "Phase 2 / Phase 3: 100% EM, CER 0, WER 0" | Not reproducible after the changes above | Use table B |
| Research paper Table I (Phase 2 EM 75%, Phase 3 BLEU 1.0) and the conclusion (CER 34.6% -> 3.1%, WER 6.8%, EM 89.3%) | These disagree with each other and with the results chapter | Use table B everywhere |
| Research paper conclusion: "proposed but not tested" | Contradicts its own results section | Reword to match what was implemented |
| "Typo modelling also includes ... overall noise probability of 25%" | True for typing mistakes (about 25% of words); vowel deletion and repetition are applied on top | Say "typing mistakes affect about 25% of words" |
| Introduction lists Telugu and Malayalam | The project covers Hindi, Kannada, Tamil, Bengali | Align with the project scope |

## 4. Things an examiner may ask

- **Why does Phase 3 add so little?** The lexicon is small, so Phase 1 already finds the
  right word most of the time. Phase 3 only decides between equally close candidates. It
  is also the slowest phase (about 170 ms).
- **Is the evaluation fair?** Partly. The test words are in the lexicon (the system cannot
  fix words it has never seen), and the noisy forms are new but come from the same noise
  generator that made the training data.
- **Why XLM-RoBERTa and not mBERT/IndicBERT?** According to its paper, its training data (CC100) includes Romanized Hindi, Bengali
  and Tamil text, so it has seen text like this.
