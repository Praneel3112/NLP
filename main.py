from src.preprocessing.cleaner import TextCleaner
from src.preprocessing.noise_generator import NoiseGenerator
from src.lang_id.detector import LanguageIdentifier
from src.detection.edit_distance import EditDistanceDetector
from src.correction.model_store import load_or_train
from src.correction.pipeline import LEXICON
from src.evaluation.metrics import EvaluationSuite
from src.emotion.detector import EmotionDetector
from src.preprocessing.slang_normalizer import SlangNormalizer


def main():
    title = "TRANSLITERATION ERROR DETECTION AND CORRECTION"
    print("=" * 96)
    print(f"{title:^96}")
    print("=" * 96)

    reference_sentences = [
        "namaste aap kaise ho",
        "nanna hesaru ravi",
        "en peyar meena",
        "amar naam suman",
    ]

    lexicon = list(LEXICON)

    cleaner = TextCleaner()
    noise_gen = NoiseGenerator(seed=42)

    clean_references = [cleaner.clean_text(s) for s in reference_sentences]
    noisy_sentences = [noise_gen.generate_noisy_text(s) for s in clean_references]

    print("STEP 1 - LANGUAGE IDENTIFICATION")
    print("-" * 96)
    lang_id = LanguageIdentifier()
    sample_tags = lang_id.segregate_code_mixed_tokens("namaste good morning aap kaise ho")
    print("  Sample: namaste good morning aap kaise ho")
    print("  Tagged tokens:")
    for tagged_token in sample_tags:
        print(f"    {tagged_token['token']:<12} -> {tagged_token['label']}")

    print("\nSTEP 2 - TEST DATA")
    print("-" * 96)
    for idx, (ref, noisy) in enumerate(zip(clean_references, noisy_sentences), 1):
        print(f"  Example {idx}")
        print(f"    Clean reference : {ref}")
        print(f"    Noisy input     : {noisy}")

    print("\nSTEP 3 - CORRECTION PHASES")
    print("-" * 96)

    print("  PHASE 1 - Rule-based edit-distance correction")
    edit_detector = EditDistanceDetector(lexicon=lexicon)
    phase1_steps = []
    edit_outputs = []
    phase1_guided_inputs = []
    for sentence in noisy_sentences:
        corrected, error_logs = edit_detector.process_sentence(sentence)
        edit_outputs.append(corrected)
        errors_by_word = {item["original"]: item for item in error_logs}
        steps = []
        guided_words = []
        for word in sentence.split():
            candidate, is_error, distance = edit_detector.detect_and_correct_word(word)
            if is_error and distance <= 1:
                guided_words.append(candidate)
                error = errors_by_word.get(word)
                candidates = (
                    ", ".join(error["candidates"])
                    if error is not None
                    else candidate
                )
                steps.append(
                    f"{word} -> {candidate} (high confidence; edit distance "
                    f"{distance}; candidates: {candidates})"
                )
            elif not is_error:
                guided_words.append(word)
                steps.append(f"{word} -> {word} (already in lexicon)")
            else:
                guided_words.append(word)
                error = errors_by_word.get(word)
                steps.append(
                    f"{word} -> {error['corrected']} "
                    f"(not passed downstream; edit distance "
                    f"{error['edit_distance']}; "
                    f"candidates: {', '.join(error['candidates'])})"
                    if error is not None
                    else f"{word} -> {word} (no high-confidence correction)"
                )
        phase1_steps.append(steps)
        phase1_guided_inputs.append(" ".join(guided_words))

    print("  PHASE 2 - Trainable character-level GRU Seq2Seq")
    # Same models as the web app: trained once (100 epochs) and reloaded from models/.
    models = load_or_train(100, on_stage=lambda stage: print(f"    {stage}"))
    rnn_model = models["gru"]
    transformer_model = models["transformer"]
    source_note = "loaded from models/" if models["loaded_from_disk"] else "trained now and saved to models/"
    print(
        f"    {len(LEXICON)}-word lexicon, {models['training_examples']} noisy-to-clean pairs, "
        f"100 epochs ({source_note}); final loss: {models['gru_loss']:.4f}."
    )
    rnn_outputs = []
    phase2_steps = []
    for sentence in phase1_guided_inputs:
        words = sentence.split()
        corrections = [rnn_model.predict_word(word) for word in words]
        rnn_outputs.append(" ".join(corrections))
        phase2_steps.append(
            [
                f"{word} -> {correction} (character encoder + autoregressive GRU decoder)"
                for word, correction in zip(words, corrections)
            ]
        )

    print("  PHASE 3 - Trainable character-level Transformer with lexicon safeguard")
    print(f"    Final loss: {models['transformer_loss']:.4f}.")
    transformer_outputs = []
    phase3_steps = []
    for sentence in phase1_guided_inputs:
        transformer_outputs.append(transformer_model.correct_sentence(sentence))
        phase3_steps.append(transformer_model.last_trace.copy())
    if transformer_model.fallback_count:
        print(
            "    The lexicon safeguard corrected out-of-lexicon outputs in "
            f"{transformer_model.fallback_count} example(s)."
        )

    print("\nPHASE OUTPUTS - SIDE-BY-SIDE COMPARISON")
    print("-" * 96)
    for idx, (reference, noisy, phase1, phase2, phase3) in enumerate(
        zip(
            clean_references,
            noisy_sentences,
            edit_outputs,
            rnn_outputs,
            transformer_outputs,
        ),
        1,
    ):
        print(f"  Example {idx}")
        print(f"    Reference : {reference}")
        print(f"    Noisy     : {noisy}")
        print(f"    Phase 1   : {phase1}")
        print(f"    Phase 2   : {phase2}")
        print(f"    Phase 3   : {phase3}")

    print("\nDETAILED PHASE TRACE - HOW EACH OUTPUT WAS PRODUCED")
    print("-" * 96)
    for idx, (sentence, guided_input, steps1, steps2, step3) in enumerate(
        zip(
            noisy_sentences,
            phase1_guided_inputs,
            phase1_steps,
            phase2_steps,
            phase3_steps,
        ),
        1,
    ):
        print(f"  Example {idx}: {sentence}")
        print("    Phase 1 - lexicon lookup and edit distance:")
        for step in steps1:
            print(f"      {step}")
        print(f"      Downstream input: {guided_input}")
        print("    Phase 2 - character-level RNN:")
        print("      Input characters -> encoder states -> greedy decoder -> output characters")
        for step in steps2:
            print(f"      {step}")
        print("    Phase 3 - Transformer generation and output validation:")
        model_output = step3["model_output"]
        if model_output is None:
            print("      Raw Transformer output: unavailable")
        else:
            print(f"      Raw Transformer output: {model_output}")
        print(f"      Decision       : {step3['decision']}")
        print(f"      Final output   : {step3['output']}")

    print("\nSTEP 4 - EVALUATION METRICS")
    print("-" * 96)
    eval_suite = EvaluationSuite()

    noisy_metrics = eval_suite.evaluate_all(clean_references, noisy_sentences)
    edit_metrics = eval_suite.evaluate_all(clean_references, edit_outputs)
    rnn_metrics = eval_suite.evaluate_all(clean_references, rnn_outputs)
    trans_metrics = eval_suite.evaluate_all(clean_references, transformer_outputs)

    table_width = 110
    print("=" * table_width)
    print(
        f"{'SYSTEM':<50} | {'CER':>8} | {'WER':>8} | {'BLEU':>8} | "
        f"{'EXACT MATCH %':>13} | {'SEM SIM':>8}"
    )
    print("=" * table_width)
    metric_rows = [
        ("Noisy input baseline", noisy_metrics),
        ("Phase 1 - Edit distance", edit_metrics),
        ("Phase 2 - Trained character-level GRU Seq2Seq", rnn_metrics),
        (
            "Phase 3 - Trained character Transformer + safeguard",
            trans_metrics,
        ),
    ]
    for label, metrics in metric_rows:
        print(
            f"{label:<50} | {metrics['CER']:>8.4f} | {metrics['WER']:>8.4f} | "
            f"{metrics['BLEU']:>8.4f} | {metrics['Exact_Match (%)']:>13.2f} | "
            f"{metrics['Semantic_Sim']:>8.4f}"
        )
    print("=" * table_width)
    print("Metric guide: lower CER/WER is better; higher BLEU, exact match, and semantic similarity are better.")

    print("\nSTEP 5 - SLANG NORMALIZATION (PHASE 0) AND EMOTION DETECTION")
    print("-" * 96)
    normalizer = SlangNormalizer()
    emotion_detector = EmotionDetector()
    emotion_metrics = emotion_detector.train()
    print(
        f"  Emotion classifier trained on {emotion_metrics['train_size']} sentences, "
        f"tested on {emotion_metrics['test_size']} unseen sentences."
    )
    for name, scores in emotion_metrics["comparison"].items():
        print(f"    {name:<11} accuracy {scores['accuracy']:.4f} | macro-F1 {scores['macro_f1']:.4f}")
    print(f"    5-fold CV accuracy (model only): {emotion_metrics['cv_accuracy_mean']:.4f}")
    for sample in ("bht gussa aa raha h 😡😡", "not happy yaar 😔", "namast aap kaise ho 😂", "love u maa ❤️"):
        phase0 = normalizer.normalize(sample)
        emotion = emotion_detector.detect(sample)
        cues = ", ".join(f"{e['token']}->{e['emotion']}" for e in emotion["evidence"]) or "none"
        print(f"  {sample}")
        print(f"    Phase 0  : {phase0['normalized']}")
        print(f"    Emotion  : {emotion['label']} ({emotion['confidence']:.0%}, {emotion['method']}, intensity {emotion['intensity']})")
        print(f"    Evidence : {cues}")
    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()