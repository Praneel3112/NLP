# Report updates: what the code now does, and what to change in the report

Everything below comes from running `python main.py` on this branch (`report-aligned`). The numbers
are reproducible: noise seeds, the GRU seed and the saved checkpoint (`models/gru_25ep.pt`) are fixed,
and Phase 1 and the XLM-RoBERTa ranker are deterministic. Any number you put in the report should be
one you can regenerate with that command.

## 1. What now matches the report

| Report / paper says | Code now |
|---|---|
| Noise: vowel deletion in words of more than two letters (p = 0.25), last letter repeated (p = 0.15), phonetic substitutions (p = 0.30), insertion / substitution / deletion / transposition | All of these (`NoiseGenerator`). Typing mistakes hit about 25% of words with 3+ letters. One extension: letters inside words can also be stretched ("yaaar"), with p = 0.05 per letter |
| LID filter with an English dictionary (NLTK) | Yes. English words skip Phases 1-3 but stay visible to Phase 3 as context. A short list of internet words (bro, wifi, whatsapp...) is added because NLTK lacks them |
| Phase 1: edit distance (tau = 2) + G2P phonetic matching | Yes. G2P ranks candidates by the edit distance between phonetic codes (the paper's eq. 4) and rescues heavily stretched words ("naaaaam" -> "naam") |
| Phase 2: 3-layer **bidirectional** GRU encoder, 1-layer GRU decoder, Bahdanau attention | Yes |
| Embedding 64, hidden 256, Adam, lr 0.001, cross-entropy with padding mask, batch 64, 25 epochs | Yes. Training is seeded, so it is reproducible |
| Phase 3: pretrained multilingual transformer for context-aware correction | `xlm-roberta-base`, used as a **candidate ranker** (not fine-tuned) |
| Distinguish genuine errors from acceptable spelling variants (accha / achha / acha) | Yes: `data/variants.csv` (62 listed variants) plus a rule that treats doubled long vowels as alternative spellings (paani / pani, saree / sari). Variants are kept exactly as typed |
| Metrics: CER, WER, BLEU, EM, precision, recall, F1, Scos (all-MiniLM-L6-v2) | All computed by `main.py`. Precision/recall/F1 are token-level (definition in `src/evaluation/metrics.py`) |
| Latency | Measured per phase on this machine's CPU |

## 2. Size of the system (replace the old figures)

* Lexicon: **1,496** Romanized Hindi, Kannada, Tamil and Bengali words (`data/lexicon.csv`; the first 13 are the original CIE3 words), plus 62 listed spelling variants.
* Phase 2 training data: **5,985** synthetic noisy-to-clean word pairs (4 noisy forms of each lexicon word, plus "ap" -> "aap"). Final loss 0.034.
* Phase 3: `xlm-roberta-base` (278M parameters), no fine-tuning.

## 3. Results for the results chapter

### The design decision behind the numbers

A spelling corrector with a small lexicon has a trade-off: the more eagerly it "fixes" words, the more
it also damages **valid words it has never seen** (names, nouns, festival names). The final system
therefore changes a word only when the difference looks like a typing slip (dropped or repeated letter,
neighbouring key, swapped letters), applies the clearest slips directly, and asks the GRU to confirm
the rest. This costs some recall on very heavily corrupted text, in exchange for safety on real text.

### A. Held-out noise benchmark (75 noisy sentences, `data/eval_sentences.csv`, "test" split)

Heavy noise: about 66% of words are corrupted. All test words are in the lexicon.

| System | CER | WER | BLEU | EM % | Prec | Recall | F1 | Scos |
|---|---|---|---|---|---|---|---|---|
| Noisy input | 0.2371 | 0.6589 | 0.0946 | 1.33 | 0.0000 | 0.0000 | 0.0000 | 0.7302 |
| Phase 1: edit distance + G2P | 0.1109 | 0.2752 | 0.3311 | **34.67** | 0.9000 | **0.5824** | **0.7071** | **0.8800** |
| Phase 2: Bi-GRU + attention | 0.1240 | 0.3023 | 0.3095 | 29.33 | 0.9020 | 0.5412 | 0.6765 | 0.8636 |
| Phase 3: XLM-RoBERTa ranker | 0.1233 | 0.2984 | 0.3098 | 29.33 | 0.9118 | 0.5471 | 0.6838 | 0.8661 |

Honest reading: on this synthetic heavy-noise set, **Phase 1 recovers the most words**. Phases 2 and 3
are more cautious (precision is the same or slightly higher, recall lower). **Do not claim that each phase
improves on the last**; the data does not show it. Claim instead: the later phases trade a little recall for
safety on text the system has not seen (table C).

What Phase 3 actually does: it only acts when a word has several equally plausible candidates, which is rare. In
150 noisy sentences it had to choose for just 5 words. It corrected three Phase 2 mistakes there
(`mila` -> `mil` twice, `nannu` -> `naanu`) and broke none, which is why its figures are a hair better than
Phase 2's. Its overall contribution is real but small; say so.

### B. Typing mistakes on known words, with unseen words typed correctly (90 sentences)

A more realistic test: a person types names and new words correctly and slips on ordinary words.

| System | CER | WER | BLEU | EM % | Prec | Recall | F1 | Scos |
|---|---|---|---|---|---|---|---|---|
| Noisy input | 0.1271 | 0.3831 | 0.2590 | 14.44 | 0.0000 | 0.0000 | 0.0000 | 0.8611 |
| Phase 1 | 0.0713 | 0.2189 | 0.5074 | 41.11 | 0.7364 | 0.5260 | 0.6136 | 0.9179 |
| Phase 2 | 0.0718 | 0.2114 | 0.4948 | 41.11 | 0.7364 | 0.5260 | 0.6136 | 0.9208 |
| Phase 3 | **0.0709** | **0.2065** | **0.4992** | **42.22** | **0.7545** | **0.5390** | **0.6288** | **0.9247** |

Here Phase 3 is the best of the three on every measure, and exact match rises from 14% to 42%.

### C. Unseen-word safety (the main new result)

Correct sentences containing valid words the lexicon has never seen. Every change is a false correction.

| Set | Sentences | Unchanged (Phase 3) | Words wrongly changed |
|---|---|---|---|
| **Blind validation** (`data/validation_sentences.csv`) | 40 (214 words) | **39 / 40** | **1**: maari -> maafi |
| Test split (`data/unseen_sentences.csv`) | 30 (134 words) | 26 / 30 | 4: priya -> periya, holi -> holo, aarti -> aati, ragi -> rahi |

How to describe these honestly (examiners may ask):

* The validation set was written before the final policy was fixed. Its **first** result, before I added the
  long-vowel spelling rule, was 36 / 40, with four changed words (maari, pyaari, saree, aapka). Three were
  alternative Romanizations of known words, which motivated the rule, so the final 39 / 40 is *after* that
  one adjustment and is no longer perfectly blind.
* The test split was looked at once while developing, to understand failure types; it is not blind either.
* The remaining failures are valid words that are **one keyboard slip away from a different real word**
  (priya / periya, holi / holo). No distance rule can separate them; only a larger dictionary or word
  frequencies could. Measured, not guessed: XLM-RoBERTa context scores overlap too much between real typos and
  valid words to serve as a gate, and the GRU's confidence is as high on its mistakes as on its correct answers.

### D. Emotion detection

| Check | Result |
|---|---|
| Held-out 84 sentences from the training file | hybrid 94.0% accuracy, macro-F1 0.94 (rules only 91.7%, model only 71.4%) |
| 42 fresh sentences written afterwards (`data/emotion_validation.csv`) | **88.1%** before the fixes below, **95.2%** after (not blind any more) |

The fixes were general: rules run on slang-expanded text, a "neutral, not sure" answer when no emoji, slang or
emotion word is found and the model is under 50% confident, and a few missing everyday words
(acha, pasand, pyaari, thanks, toot...). The rule scores are optimistic, because the same people wrote the
dictionaries and the first test sentences; the 88.1% blind figure is the fairer one.

### E. Latency (CPU, one thread, per sentence)

Phase 1 2.2 ms, Phase 2 3.5 ms, Phase 3 5.7 ms, total about 11 ms. Phase 3 is cheap because XLM-RoBERTa only
runs when a word has several plausible candidates. (An ambiguous word costs roughly 100-200 ms.)

### F. The four demo sentences (the table the report currently shows)

Four sentences cannot support conclusions. With the harder noise, Phase 1 / 2 / 3 reach 0% / 0% / 0% exact
match and F1 0.20 / 0.55 / 0.55. Prefer table A, B or C.

## 4. Report statements that the code cannot support

Change or remove these, or an examiner may ask for evidence you do not have.

| In the report / paper | Problem | Suggested replacement |
|---|---|---|
| Phase 3 "fine-tuned IndicXlit / XLM-RoBERTa", "multilingual seq2seq Transformer" | Phase 3 is XLM-RoBERTa used without fine-tuning, to rank candidates. IndicXlit does Roman to native-script transliteration, a different task | "A pretrained XLM-RoBERTa-base model scores candidate corrections in the context of the sentence" |
| "AdamW, lr 5e-5, weight decay 0.01", "batch size 16" for Phase 3 | Nothing is fine-tuned | Remove, or say "no fine-tuning" |
| IndicGLUE / IndicXlit corpus, **500,000** pairs, 80:10:10 split | The project trains on **5,985** pairs generated from a 1,496-word lexicon and tests on generated sentences | "5,985 synthetic noisy-to-clean word pairs generated from a 1,496-word lexicon; evaluation on held-out noisy sentences (table A), mixed sentences (table B) and unseen-word sentences (table C)" |
| "150,000-word dictionary", "under 1.8 ms per word" | The lexicon has 1,496 words. The English list (NLTK, about 234,000 words) is used by language ID | "1,496-word Romanized Indic lexicon; English filter from the NLTK words corpus" |
| Phase 2 "handles out-of-dictionary / unseen words" | The GRU's answer is only accepted when it is a lexicon word that a typing slip could explain. Unseen words are deliberately left as typed | "Phase 2 confirms corrections for words close to the lexicon; words outside the lexicon are kept unchanged by design" |
| LID "96.2% accuracy / precision" | Never measured | Remove, or measure on a labelled token set |
| RTX 3090, CUDA 11.8, TensorRT, FP16, Intel i7-12700K, "42.6 ms" latency | The code runs on CPU; no GPU code exists | Report the measured CPU latency above and your real hardware |
| "Python 3.10 / PyTorch 2.0 / Ubuntu 22.04" | The project runs on Python 3.12, PyTorch CPU, Windows 11 | Use your real environment |
| Results table "Phase 2 / Phase 3: 100% EM, CER 0, WER 0" | Not reproducible; it was 4 easy sentences with a 13-word lexicon | Use tables A, B, C |
| Paper Table I (Phase 2 EM 75%, Phase 3 BLEU 1.0) and conclusion (CER 34.6% -> 3.1%, WER 6.8%, EM 89.3%) | They disagree with each other and with the results chapter | Use the tables above everywhere |
| Paper conclusion: "proposed but not tested" | Contradicts its own results section | Reword to what was implemented |
| "Repeats the last character" formula | True for the last letter; the code also stretches letters inside words (an extension) | Mention both |
| Introduction lists Telugu and Malayalam | The project covers Hindi, Kannada, Tamil, Bengali | Align with the scope |

## 5. Known limits (say these before you are asked)

* A valid word that is one typing slip from a **different** lexicon word may be changed (priya -> periya).
* A valid alternative spelling that is **not** in `data/variants.csv` and does not follow the long-vowel rule may be changed.
* Words outside the lexicon are never *fixed*, only preserved. Add words to `data/lexicon.csv` to extend coverage (the GRU retrains automatically, about 7 minutes).
* Heavily corrupted words (3+ errors) are mostly not recovered.
* Sarcasm is not detected by the emotion model.
* The test words come from the same generator as the training data, and every test word is in the lexicon, so table A is optimistic about *known* words.

## 6. Things an examiner may ask

* **Why do Phases 2 and 3 not beat Phase 1 on table A?** Because they are deliberately cautious. A corrector that
  maximises recall on a small lexicon rewrites valid new words (table C). The final design chooses safety.
* **Why XLM-RoBERTa and not mBERT / IndicBERT?** According to its paper, its training data (CC100) includes
  Romanized Hindi, Bengali and Tamil text, so it has seen text like this.
* **How do you know it works on unseen words?** Table C: 39 of 40 sentences unchanged on a set written before
  the policy was fixed, with the caveats listed above.
