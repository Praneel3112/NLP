"""Train the Phase 2 Bi-GRU once and reload it from disk afterwards.

The checkpoint lives in ``models/`` and carries a fingerprint of everything that
affects training (training pairs, epochs, architecture sizes, seed). If the lexicon,
the noise engine or any setting changes, the fingerprint no longer matches and the
model retrains. Phase 3 uses a pretrained transformer and needs no training here.
"""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from src.correction.pipeline import LEXICON, build_training_pairs
from src.correction.rnn_baseline import EMBEDDING_DIM, HIDDEN_DIM, Seq2SeqRNN

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
CHECKPOINT_VERSION = 2  # 2 = bidirectional encoder, batched training
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
SEED = 42


def _fingerprint(settings: dict[str, Any], sources: list[str], targets: list[str]) -> str:
    payload = json.dumps(
        {"version": CHECKPOINT_VERSION, "settings": settings, "sources": sources, "targets": targets},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _vocab_list(index_to_char: dict[int, str]) -> list[str]:
    return [index_to_char[index] for index in range(len(index_to_char))]


def _load_gru(path: Path, fingerprint: str) -> tuple[Seq2SeqRNN, float] | None:
    if not path.is_file():
        return None
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data.get("fingerprint") != fingerprint:
        return None
    model = Seq2SeqRNN(device="cpu", lexicon=list(LEXICON))
    model._reset_model_for_vocab(data["vocab"])
    model.load_state_dict(data["state_dict"])
    model.eval()
    return model, float(data["loss"])


def load_or_train(
    epochs: int,
    on_stage: Callable[[str], None] = lambda _stage: None,
    models_dir: Path = MODELS_DIR,
) -> dict[str, Any]:
    """Return the trained GRU, training it only when no valid checkpoint exists."""
    sources, targets = build_training_pairs()
    settings = {
        "model": "bigru-bahdanau",
        "epochs": epochs,
        "lr": LEARNING_RATE,
        "teacher_forcing": 1.0,
        "batch_size": BATCH_SIZE,
        "embedding_dim": EMBEDDING_DIM,
        "hidden_dim": HIDDEN_DIM,
        "seed": SEED,
    }
    fingerprint = _fingerprint(settings, sources, targets)
    path = models_dir / f"gru_{epochs}ep.pt"

    loaded = _load_gru(path, fingerprint)
    if loaded is not None:
        gru, loss = loaded
        from_disk = True
    else:
        on_stage(
            f"Training Phase 2 · Bi-GRU with attention ({epochs} epochs, {len(sources):,} examples; "
            "first run only, a few minutes)"
        )
        gru = Seq2SeqRNN(device="cpu", lexicon=list(LEXICON))
        loss = gru.train_on_pairs(
            sources,
            targets,
            epochs=epochs,
            learning_rate=LEARNING_RATE,
            teacher_forcing_ratio=1.0,
            batch_size=BATCH_SIZE,
            seed=SEED,
        )
        models_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"fingerprint": fingerprint, "vocab": _vocab_list(gru.idx2char), "state_dict": gru.state_dict(), "loss": loss},
            path,
        )
        from_disk = False

    return {
        "gru": gru,
        "gru_loss": loss,
        "training_examples": len(sources),
        "loaded_from_disk": from_disk,
    }
