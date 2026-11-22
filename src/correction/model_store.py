"""Train the Phase 2/3 models once and reload them from disk afterwards.

Checkpoints live in ``models/`` and carry a fingerprint of everything that
affects training (training pairs, epochs, learning settings). If the lexicon or
any setting changes, the fingerprint no longer matches and the models retrain.
"""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from src.correction.pipeline import LEXICON, build_training_pairs
from src.correction.rnn_baseline import Seq2SeqRNN
from src.correction.transformer_models import TransformerCorrector, _CharacterTransformer

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
CHECKPOINT_VERSION = 1


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
    return model, float(data["loss"])


def _load_transformer(path: Path, fingerprint: str) -> tuple[TransformerCorrector, float] | None:
    if not path.is_file():
        return None
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data.get("fingerprint") != fingerprint:
        return None
    corrector = TransformerCorrector(device="cpu", lexicon=list(LEXICON))
    vocab = data["vocab"]
    corrector.char2idx = {token: index for index, token in enumerate(vocab)}
    corrector.idx2char = dict(enumerate(vocab))
    corrector.model = _CharacterTransformer(vocab_size=len(vocab), pad_idx=corrector.char2idx["<pad>"])
    corrector.model.load_state_dict(data["state_dict"])
    corrector.model.eval()
    return corrector, float(data["loss"])


def load_or_train(
    epochs: int,
    on_stage: Callable[[str], None] = lambda _stage: None,
    models_dir: Path = MODELS_DIR,
) -> dict[str, Any]:
    """Return trained GRU and Transformer models, training only when no valid checkpoint exists."""
    sources, targets = build_training_pairs()
    gru_settings = {"model": "gru", "epochs": epochs, "lr": 1e-3, "teacher_forcing": 1.0}
    transformer_settings = {"model": "transformer", "epochs": epochs, "lr": 1e-3, "batch_size": 16}
    gru_key = _fingerprint(gru_settings, sources, targets)
    transformer_key = _fingerprint(transformer_settings, sources, targets)
    gru_path = models_dir / f"gru_{epochs}ep.pt"
    transformer_path = models_dir / f"transformer_{epochs}ep.pt"

    loaded_gru = _load_gru(gru_path, gru_key)
    if loaded_gru is None:
        on_stage(
            f"Training Phase 2 · character-level GRU ({epochs} epochs, {len(sources):,} examples; "
            "first run only, about 15 minutes)"
        )
        gru = Seq2SeqRNN(device="cpu", lexicon=list(LEXICON))
        gru_loss = gru.train_on_pairs(sources, targets, epochs=epochs, learning_rate=1e-3, teacher_forcing_ratio=1.0)
        models_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"fingerprint": gru_key, "vocab": _vocab_list(gru.idx2char), "state_dict": gru.state_dict(), "loss": gru_loss},
            gru_path,
        )
        gru_cached = False
    else:
        gru, gru_loss = loaded_gru
        gru_cached = True

    loaded_transformer = _load_transformer(transformer_path, transformer_key)
    if loaded_transformer is None:
        on_stage(f"Training Phase 3 · character Transformer ({epochs} epochs, {len(sources):,} examples)")
        transformer = TransformerCorrector(device="cpu", lexicon=list(LEXICON))
        transformer_loss = transformer.train_on_pairs(sources, targets, epochs=epochs, learning_rate=1e-3, batch_size=16)
        models_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "fingerprint": transformer_key,
                "vocab": _vocab_list(transformer.idx2char),
                "state_dict": transformer.model.state_dict(),
                "loss": transformer_loss,
            },
            transformer_path,
        )
        transformer_cached = False
    else:
        transformer, transformer_loss = loaded_transformer
        transformer_cached = True

    return {
        "gru": gru,
        "transformer": transformer,
        "gru_loss": gru_loss,
        "transformer_loss": transformer_loss,
        "training_examples": len(sources),
        "loaded_from_disk": gru_cached and transformer_cached,
    }
