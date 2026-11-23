"""Held-out benchmark for the correction phases (used by ``main.py`` and the tests)."""

import csv
import time
from pathlib import Path
from typing import Any

from src.evaluation.metrics import EvaluationSuite
from src.preprocessing.noise_generator import NoiseGenerator

EVAL_PATH = Path(__file__).resolve().parents[2] / "data" / "eval_sentences.csv"
NOISE_SEED_BASE = 5000  # training noise uses seeds 1000-5000, so these noisy forms are new
VARIANTS_PER_SENTENCE = 3
PHASES = ("phase1", "phase2", "phase3")


def load_eval_sentences(split: str | None = None, path: Path = EVAL_PATH) -> list[str]:
    """Clean evaluation sentences, optionally only the ``"tune"`` or ``"test"`` half."""
    with path.open(encoding="utf-8", newline="") as handle:
        return [row["sentence"] for row in csv.DictReader(handle) if split in (None, row["split"])]


def noisy_pairs(sentences: list[str], variants: int = VARIANTS_PER_SENTENCE) -> list[tuple[str, str]]:
    """(clean, noisy) pairs; each sentence is corrupted ``variants`` times with fresh seeds."""
    pairs = []
    for index, sentence in enumerate(sentences):
        for variant in range(variants):
            generator = NoiseGenerator(seed=NOISE_SEED_BASE + index * variants + variant)
            pairs.append((sentence, generator.generate_noisy_text(sentence)))
    return pairs


def run_pipeline(service: Any, pairs: list[tuple[str, str]]) -> dict[str, Any]:
    """Run Phases 1-3 over every noisy sentence; return outputs and per-phase timings."""
    outputs: dict[str, list[str]] = {phase: [] for phase in PHASES}
    totals = {phase: 0.0 for phase in PHASES}
    total_ms = 0.0
    for _, noisy in pairs:
        started = time.perf_counter()
        result = service.run_phases(noisy)
        total_ms += (time.perf_counter() - started) * 1000
        for phase in PHASES:
            outputs[phase].append(result[phase])
            totals[phase] += result["timings_ms"][phase]
    count = max(len(pairs), 1)
    return {
        "outputs": outputs,
        "latency_ms": {**{phase: totals[phase] / count for phase in PHASES}, "total": total_ms / count},
    }


def score(suite: EvaluationSuite, pairs: list[tuple[str, str]], outputs: dict[str, list[str]]) -> dict[str, dict]:
    """Metrics for the noisy baseline and each phase."""
    references = [clean for clean, _ in pairs]
    noisy = [noisy for _, noisy in pairs]
    rows = {"Noisy input": suite.evaluate_all(references, noisy, inputs=noisy)}
    for phase in PHASES:
        rows[phase] = suite.evaluate_all(references, outputs[phase], inputs=noisy)
    return rows
