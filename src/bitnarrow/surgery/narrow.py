from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bitnarrow.data import PromptSource
from bitnarrow.models import decoder_layers
from bitnarrow.surgery.common import collect_last_token_hidden

WRITERS = {"o_proj", "down_proj", "out_proj", "fc2", "dense_4h_to_h", "w2", "wo"}
READERS = {"q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "fc1", "dense_h_to_4h", "w1", "w3", "wi"}
DEFAULT_TARGETS = ["o_proj", "down_proj", "gate_proj", "up_proj"]


@dataclass
class NarrowConfig:
    positive: PromptSource
    negative: PromptSource
    winsor_quantile: float = 0.95
    layer: int | str = "auto"
    layer_window: tuple[float, float] = (0.2, 0.8)
    band: str | list[int] = "all"
    band_threshold: float = 0.5
    strength: float | list[float] = 1.0
    targets: list[str] = field(default_factory=lambda: list(DEFAULT_TARGETS))
    edit_embeddings: bool = False
    batch_size: int = 8

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> NarrowConfig:
        params = dict(params)
        for key in ("positive", "negative"):
            if key not in params:
                raise ValueError(f"narrow params need {key!r} prompt source")
            params[key] = PromptSource.from_dict(params[key])
        if "layer_window" in params:
            params["layer_window"] = tuple(params["layer_window"])
        cfg = cls(**params)
        bad = set(cfg.targets) - WRITERS - READERS
        if bad:
            raise ValueError(f"unknown target modules {sorted(bad)}")
        if not 0.5 <= cfg.winsor_quantile <= 1.0:
            raise ValueError("winsor_quantile must be in [0.5, 1.0]")
        return cfg

    def strengths(self) -> list[float]:
        return [float(s) for s in self.strength] if isinstance(self.strength, list) else [float(self.strength)]


@dataclass
class Direction:
    vector: Any
    layer: int
    band: list[int]
    scores: list[float]
    band_cosines: list[float]


def winsorized_mean(x, quantile: float):
    import torch

    if quantile >= 1.0:
        return x.mean(dim=0)
    cap = torch.quantile(x.abs(), quantile, dim=0)
    return torch.maximum(torch.minimum(x, cap), -cap).mean(dim=0)


def layer_directions(pos, neg, quantile: float):
    import torch

    mp = torch.stack([winsorized_mean(pos[i], quantile) for i in range(pos.shape[0])])
    mn = torch.stack([winsorized_mean(neg[i], quantile) for i in range(neg.shape[0])])
    diff = mp - mn
    norms = diff.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    dirs = diff / norms
    pp = torch.einsum("lnd,ld->ln", pos, dirs)
    pn = torch.einsum("lnd,ld->ln", neg, dirs)
    pooled = ((pp.var(dim=1, unbiased=False) + pn.var(dim=1, unbiased=False)) / 2).sqrt().clamp_min(1e-6)
    scores = (pp.mean(dim=1) - pn.mean(dim=1)) / pooled
    return dirs, scores


def select_layer(scores, n_layers: int, window: tuple[float, float]) -> int:
    lo = max(1, int(round(window[0] * n_layers)))
    hi = min(n_layers, max(lo, int(round(window[1] * n_layers))))
    sub = scores[lo : hi + 1]
    return lo + int(sub.argmax())


def select_band(dirs, layer: int, n_layers: int, mode: str | list[int], threshold: float) -> tuple[list[int], list[float]]:
    cos = [float(dirs[i + 1] @ dirs[layer]) for i in range(n_layers)]
    if mode == "all":
        return list(range(n_layers)), cos
    if isinstance(mode, list):
        a, b = int(mode[0]), int(mode[1])
        if not 0 <= a <= b < n_layers:
            raise ValueError(f"band {mode} outside [0, {n_layers - 1}]")
        return list(range(a, b + 1)), cos
    if mode != "auto":
        raise ValueError("band must be all, auto, or [start, end]")
    anchor = min(max(layer - 1, 0), n_layers - 1)
    start = end = anchor
    while start - 1 >= 0 and cos[start - 1] >= threshold:
        start -= 1
    while end + 1 < n_layers and cos[end + 1] >= threshold:
        end += 1
    return list(range(start, end + 1)), cos


def find_direction(model, tokenizer, cfg: NarrowConfig, chat: bool = True) -> Direction:
    layers, _ = decoder_layers(model)
    n_layers = len(layers)
    pos = collect_last_token_hidden(model, tokenizer, cfg.positive.load(), cfg.batch_size, chat)
    neg = collect_last_token_hidden(model, tokenizer, cfg.negative.load(), cfg.batch_size, chat)
    dirs, scores = layer_directions(pos, neg, cfg.winsor_quantile)
    if cfg.layer == "auto":
        layer = select_layer(scores, n_layers, cfg.layer_window)
    else:
        layer = int(cfg.layer)
        if not 1 <= layer <= n_layers:
            raise ValueError(f"layer must be in [1, {n_layers}] (hidden state index)")
    band, cos = select_band(dirs, layer, n_layers, cfg.band, cfg.band_threshold)
    return Direction(
        vector=dirs[layer].clone(),
        layer=layer,
        band=band,
        scores=[float(s) for s in scores],
        band_cosines=cos,
    )


def project_weight(weight, r, role: str, strength: float):
    w = weight.float()
    r = r.to(device=w.device, dtype=w.dtype)
    if role == "writer":
        return w - strength * r.unsqueeze(1) @ (r.unsqueeze(0) @ w)
    return w - strength * (w @ r).unsqueeze(1) @ r.unsqueeze(0)


def _role(leaf: str) -> str:
    return "writer" if leaf in WRITERS else "reader"


def build_patch(model, direction: Direction, cfg: NarrowConfig, strength: float) -> dict[str, Any]:
    import torch

    layers, path = decoder_layers(model)
    r = direction.vector.float()
    hidden = r.shape[0]
    patch: dict[str, Any] = {}
    with torch.no_grad():
        for i in direction.band:
            for sub, module in layers[i].named_modules():
                leaf = sub.rsplit(".", 1)[-1]
                if leaf not in cfg.targets or not isinstance(module, torch.nn.Linear):
                    continue
                role = _role(leaf)
                w = module.weight
                if role == "writer" and w.shape[0] != hidden:
                    continue
                if role == "reader" and w.shape[1] != hidden:
                    continue
                prefix = f"{path}.{i}.{sub}"
                patch[f"{prefix}.weight"] = project_weight(w, r, role, strength).to(w.dtype).cpu()
                if role == "writer" and module.bias is not None:
                    b = module.bias.float()
                    rb = r.to(b.device)
                    patch[f"{prefix}.bias"] = (b - strength * rb * (rb @ b)).to(module.bias.dtype).cpu()
        if cfg.edit_embeddings:
            emb = model.get_input_embeddings()
            name = next(n for n, p in model.named_parameters() if p is emb.weight)
            e = emb.weight.float()
            re = r.to(e.device)
            patch[name] = (e - strength * (e @ re).unsqueeze(1) @ re.unsqueeze(0)).to(emb.weight.dtype).cpu()
    if not patch:
        raise RuntimeError(f"no target modules matched {cfg.targets} in band {direction.band}")
    return patch


def patch_meta(direction: Direction, cfg: NarrowConfig, strength: float, base_model: str, base_revision: str | None):
    return {
        "method": "narrow",
        "base_model": base_model,
        "base_revision": base_revision,
        "strength": strength,
        "direction_layer": direction.layer,
        "band": direction.band,
        "targets": cfg.targets,
        "winsor_quantile": cfg.winsor_quantile,
        "edit_embeddings": cfg.edit_embeddings,
        "layer_scores": direction.scores,
        "band_cosines": direction.band_cosines,
    }
