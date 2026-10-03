from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from bitnarrow.data import PromptSource
from bitnarrow.models import decoder_layers
from bitnarrow.surgery.common import collect_last_token_hidden

MODES = {"block", "layerwise"}


@dataclass
class PruneConfig:
    calibration: PromptSource
    n_remove: int
    mode: str = "block"
    protect_last: bool = True
    batch_size: int = 8

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> PruneConfig:
        params = dict(params)
        if "calibration" not in params:
            raise ValueError("prune params need a 'calibration' prompt source")
        params["calibration"] = PromptSource.from_dict(params["calibration"])
        cfg = cls(**params)
        if cfg.mode not in MODES:
            raise ValueError(f"mode must be one of {sorted(MODES)}")
        if cfg.n_remove < 1:
            raise ValueError("n_remove must be at least 1")
        return cfg


def angular_distance(a, b) -> float:
    import torch

    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1).clamp(-1.0, 1.0)
    return float((torch.arccos(cos) / math.pi).mean())


def score_blocks(hidden, n: int, protect_last: bool) -> list[tuple[int, float]]:
    n_layers = hidden.shape[0] - 1
    last_start = n_layers - n - (1 if protect_last else 0)
    if last_start < 0:
        raise ValueError(f"cannot remove {n} of {n_layers} layers")
    return [(s, angular_distance(hidden[s], hidden[s + n])) for s in range(last_start + 1)]


def score_layers(hidden, protect_last: bool) -> list[tuple[int, float]]:
    n_layers = hidden.shape[0] - 1
    stop = n_layers - (1 if protect_last else 0)
    return [(i, angular_distance(hidden[i], hidden[i + 1])) for i in range(stop)]


def choose(model, tokenizer, cfg: PruneConfig, chat: bool = True) -> dict[str, Any]:
    hidden = collect_last_token_hidden(model, tokenizer, cfg.calibration.load(), cfg.batch_size, chat)
    n_layers = hidden.shape[0] - 1
    if cfg.n_remove >= n_layers:
        raise ValueError(f"n_remove {cfg.n_remove} must be below layer count {n_layers}")
    if cfg.mode == "block":
        scores = score_blocks(hidden, cfg.n_remove, cfg.protect_last)
        start, _ = min(scores, key=lambda t: t[1])
        remove = list(range(start, start + cfg.n_remove))
    else:
        scores = score_layers(hidden, cfg.protect_last)
        if cfg.n_remove > len(scores):
            raise ValueError("n_remove exceeds removable layers")
        remove = sorted(i for i, _ in sorted(scores, key=lambda t: t[1])[: cfg.n_remove])
    return {"remove": remove, "scores": scores, "n_layers": n_layers, "mode": cfg.mode}


def remove_layers(model, indices: list[int]) -> int:
    import torch

    layers, path = decoder_layers(model)
    drop = set(indices)
    kept = [layer for i, layer in enumerate(layers) if i not in drop]
    parent = model
    for part in path.split(".")[:-1]:
        parent = getattr(parent, part)
    setattr(parent, path.split(".")[-1], torch.nn.ModuleList(kept))
    for new_idx, layer in enumerate(kept):
        for module in layer.modules():
            if hasattr(module, "layer_idx"):
                module.layer_idx = new_idx
    cfg = model.config
    old = cfg.num_hidden_layers
    cfg.num_hidden_layers = len(kept)
    layer_types = getattr(cfg, "layer_types", None)
    if isinstance(layer_types, list) and len(layer_types) == old:
        cfg.layer_types = [t for i, t in enumerate(layer_types) if i not in drop]
    if hasattr(cfg, "max_window_layers") and isinstance(cfg.max_window_layers, int):
        cfg.max_window_layers = min(cfg.max_window_layers, len(kept))
    return len(kept)
