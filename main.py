"""Command-line run of the whole project: every phase, the metrics, and emotion detection."""

import warnings

from src.correction.pipeline import LEXICON, VARIANTS
from src.correction.service import TRAINING_EPOCHS, CorrectionService
from src.emotion.detector import EmotionDetector
from src.evaluation import benchmark
from src.evaluation.metrics import EvaluationSuite
from src.lang_id.detector import LanguageIdentifier
from src.preprocessing.cleaner import TextCleaner
from src.preprocessing.noise_generator import NoiseGenerator
from src.preprocessing.slang_normalizer import SlangNormalizer

warnings.filterwarnings("ignore", message=".*nested tensors.*")

RULE = "-" * 110


def print_metric_table(title: str, rows: dict[str, dict]) -> None:
    print(f"\n  {title}")
    print("  " + "=" * 124)
    print(
        f"  {'SYSTEM':<34} | {'CER':>7} | {'WER':>7} | {'BLEU':>7} | {'EM %':>7} | "
        f"{'PREC':>7} | {'RECALL':>7} | {'F1':>7} | {'SEM SIM':>7}"
    )
    print("  " + "=" * 124)
    for label, m in rows.items():
        print(
            f"  {label:<34} | {m['CER']:>7.4f} | {m['WER']:>7.4f} | {m['BLEU']:>7.4f} | "
            f"{m['Exact_Match (%)']:>7.2f} | {m['Precision']:>7.4f} | {m['Recall']:>7.4f} | "
            f"{m['F1']:>7.4f} | {m['Semantic_Sim']:>7.4f}"
        )
    print("  " + "=" * 124)


def main():
    title = "TRANSLITERATION ERROR DETECTION AND CORRECTION"
    print("=" * 110)
    print(f"{title:^110}")
    print("=" * 110)

    reference_sentences = [
        "namaste aap kaise ho",
        "nanna hesaru ravi",
        "en peyar meena",
        "amar naam suman",
    ]

    cleaner = TextCleaner()
    noise_gen = NoiseGenerator(seed=42)
    clean_references = [cleaner.clean_text(s) for s in reference_sentences]
    noisy_sentences = [noise_gen.generate_noisy_text(s) for s in clean_references]

    print("STEP 1 - LANGUAGE IDENTIFICATION")
    print(RULE)
    lang_id = LanguageIdentifier()
    sample = "namaste good morning aap kaise ho"
    print(f"  Sample: {sample}")
    print("  Tagged tokens:")
    for tagged_token in lang_id.segregate_code_mixed_tokens(sample):
        print(f"    {tagged_token['token']:<12} -> {tagged_token['label']}")

    print("\nSTEP 2 - TEST DATA")
    print(RULE)
    for idx, (ref, noisy) in enumerate(zip(clean_references, noisy_sentences), 1):
        print(f"  Example {idx}")
        print(f"    Clean reference : {ref}")
        print(f"    Noisy input     : {noisy}")

    print("\nSTEP 3 - CORRECTION PHASES")
    print(RULE)
    service = CorrectionService()
    print(f"  Lexicon: {len(LEXICON)} Romanized Hindi/Kannada/Tamil/Bengali words (data/lexicon.csv) + {len(VARIANTS)} accepted spelling variants.")
    service.load_correction_models()
    status = service.get_status()
    source = "loaded from models/" if status["models_loaded_from_disk"] else "trained now and saved to models/"
    print(
        f"  PHASE 1 - edit distance (<= 2) + phonetic G2P matching against the lexicon\n"
        f"  PHASE 2 - 3-layer Bi-GRU with Bahdanau attention: {status['training_examples']} noisy-to-clean "
        f"pairs, {TRAINING_EPOCHS} epochs, batch 64 ({source}; final loss {status['phase2_loss']})\n"
        f"  PHASE 3 - XLM-RoBERTa context ranking: {status['phase3_model']}"
    )
    results = [service.run_phases(noisy) for noisy in noisy_sentences]

    print("\nPHASE OUTPUTS - SIDE-BY-SIDE COMPARISON")
    print(RULE)
    for idx, (reference, noisy, result) in enumerate(zip(clean_references, noisy_sentences, results), 1):
        print(f"  Example {idx}")
        print(f"    Reference : {reference}")
        print(f"    Noisy     : {noisy}")
        print(f"    Phase 1   : {result['phase1']}")
        print(f"    Phase 2   : {result['phase2']}")
        print(f"    Phase 3   : {result['phase3']}")

    print("\nDETAILED PHASE TRACE - HOW EACH OUTPUT WAS PRODUCED")
    print(RULE)
    for idx, (noisy, result) in enumerate(zip(noisy_sentences, results), 1):
        print(f"  Example {idx}: {noisy}")
        print("    Phase 1 - lexicon lookup, edit distance and G2P:")
        for step in result["phase1_steps"]:
            print(f"      {step['original']} -> {step['corrected']} ({step['decision']}; edit distance {step['distance']})")
        print("    Phase 2 - Bi-GRU:")
        for step in result["phase2_steps"]:
            print(f"      {step['original']} -> {step['model_output']} ({step['decision']})")
        print("    Phase 3 - XLM-RoBERTa ranking:")
        print(f"      {result['phase3_trace']['decision']}")
        for ranking in result["phase3_trace"].get("rankings", []):
            listed = ", ".join(f"{c['word']} ({c['score']:.1f})" for c in ranking["candidates"])
            print(f"      {ranking['token']}: {listed}  ->  {ranking['chosen']}")

    print("\nSTEP 4 - EVALUATION METRICS")
    print(RULE)
    suite = EvaluationSuite()
    demo_pairs = list(zip(clean_references, noisy_sentences))
    demo_outputs = {phase: [r[phase] for r in results] for phase in benchmark.PHASES}
    labels = {
        "Noisy input": "Noisy input baseline",
        "phase1": "Phase 1 - edit distance + G2P",
        "phase2": "Phase 2 - Bi-GRU + attention",
        "phase3": "Phase 3 - XLM-RoBERTa ranker",
    }
    demo_scores = benchmark.score(suite, demo_pairs, demo_outputs)
    print_metric_table(
        f"A. The {len(demo_pairs)} demo sentences above",
        {labels[key]: value for key, value in demo_scores.items()},
    )

    test_sentences = benchmark.load_eval_sentences("test")
    test_pairs = benchmark.noisy_pairs(test_sentences)
    run = benchmark.run_pipeline(service, test_pairs)
    test_scores = benchmark.score(suite, test_pairs, run["outputs"])
    print_metric_table(
        f"B. Held-out benchmark: {len(test_sentences)} sentences x {benchmark.VARIANTS_PER_SENTENCE} fresh noisy "
        f"versions = {len(test_pairs)} test sentences (data/eval_sentences.csv, 'test' split)",
        {labels[key]: value for key, value in test_scores.items()},
    )
    latency = run["latency_ms"]
    print(
        f"\n  Latency on this machine (CPU, {len(test_pairs)} sentences): "
        f"Phase 1 {latency['phase1']:.1f} ms | Phase 2 {latency['phase2']:.1f} ms | "
        f"Phase 3 {latency['phase3']:.1f} ms | total {latency['total']:.1f} ms per sentence"
    )
    # ---- unseen words: valid words the lexicon has never seen must not be "corrected" -------------
    unseen_test = benchmark.load_unseen_sentences("test")
    kept = benchmark.preservation(service, unseen_test)
    print(
        f"\n  C. Unseen-word safety: {kept['sentences']} correct sentences ({kept['words']} words) containing names, "
        "nouns, festivals and English words\n     that the lexicon has never seen (data/unseen_sentences.csv, 'test' split). "
        "Every change is a false correction."
    )
    for phase in benchmark.PHASES:
        count = kept["unchanged_sentences"][phase]
        wrong = kept["false_corrections"][phase]
        shown = ", ".join(f"{a} -> {b}" for a, b in wrong) or "none"
        print(f"     {labels[phase]:<34}: {count}/{kept['sentences']} sentences unchanged | false corrections: {shown}")

    blind = benchmark.load_validation_sentences()
    blind_result = benchmark.preservation(service, blind)
    print(
        f"\n  C2. Blind validation: {blind_result['sentences']} more correct sentences ({blind_result['words']} words) "
        "with new unseen words (data/validation_sentences.csv).\n     Written before the final policy was fixed and never "
        "used to tune anything: the honest estimate."
    )
    for phase in benchmark.PHASES:
        count = blind_result["unchanged_sentences"][phase]
        wrong = blind_result["false_corrections"][phase]
        shown = ", ".join(f"{a} -> {b}" for a, b in wrong) or "none"
        print(f"     {labels[phase]:<34}: {count}/{blind_result['sentences']} sentences unchanged | false corrections: {shown}")

    mixed_pairs = benchmark.mixed_noise_pairs(unseen_test, set(LEXICON), set(VARIANTS))
    mixed_run = benchmark.run_pipeline(service, mixed_pairs)
    mixed_scores = benchmark.score(suite, mixed_pairs, mixed_run["outputs"])
    print_metric_table(
        f"D. Typing mistakes on known words only, unseen words typed correctly ({len(mixed_pairs)} sentences)",
        {labels[key]: value for key, value in mixed_scores.items()},
    )

    print(
        "\n  Metric guide: lower CER/WER is better; higher BLEU, exact match, precision, recall, "
        "F1 and semantic similarity are better."
    )
    print(
        "  Precision/recall/F1 are token-level: a true positive is a wrong input word that the system "
        "corrected exactly;\n  a false positive is a word the system changed into something wrong."
    )

    print("\nSTEP 5 - SLANG NORMALIZATION (PHASE 0) AND EMOTION DETECTION")
    print(RULE)
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
