from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from bitnarrow.models import PATCH_META, PATCH_WEIGHTS

LAYER_RE = re.compile(r"\.(?:layers|h)\.(\d+)\.")


class TensorSource:
    def __init__(self, files: dict[str, Path], label: str):
        self._files = files
        self.label = label
        self._handles: dict[Path, Any] = {}

    @classmethod
    def open(cls, ref: str | Path) -> TensorSource:
        path = Path(ref)
        if not path.exists():
            from huggingface_hub import snapshot_download

            path = Path(snapshot_download(str(ref), allow_patterns=["*.safetensors", "*.json"]))
        if path.is_file():
            shards = [path]
        elif (path / PATCH_WEIGHTS).exists():
            shards = [path / PATCH_WEIGHTS]
        else:
            shards = sorted(path.glob("*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"no safetensors found at {ref}")
        from safetensors import safe_open

        files: dict[str, Path] = {}
        for shard in shards:
            with safe_open(str(shard), framework="pt") as f:
                for key in f.keys():
                    files[key] = shard
        return cls(files, str(ref))

    def keys(self) -> list[str]:
        return list(self._files)

    def get(self, key: str):
        from safetensors import safe_open

        shard = self._files[key]
        if shard not in self._handles:
            self._handles[shard] = safe_open(str(shard), framework="pt")
        return self._handles[shard].get_tensor(key)

    def __contains__(self, key: str) -> bool:
        return key in self._files


@dataclass
class TensorDiff:
    name: str
    shape: list[int]
    rel_change: float
    cosine: float
    frac_changed: float
    top1_energy: float | None = None


@dataclass
class DiffReport:
    base: str
    other: str
    tensors: list[TensorDiff] = field(default_factory=list)
    only_in_base: list[str] = field(default_factory=list)
    only_in_other: list[str] = field(default_factory=list)
    shape_mismatch: list[str] = field(default_factory=list)
    unchanged: int = 0

    def by_layer(self) -> dict[str, float]:
        acc: dict[str, list[float]] = defaultdict(list)
        for t in self.tensors:
            m = LAYER_RE.search(t.name)
            acc[m.group(1) if m else "other"].append(t.rel_change)
        return {k: max(v) for k, v in sorted(acc.items(), key=lambda kv: (kv[0] == "other", int(kv[0]) if kv[0].isdigit() else 0))}

    def by_module(self) -> dict[str, float]:
        acc: dict[str, list[float]] = defaultdict(list)
        for t in self.tensors:
            parts = t.name.split(".")
            leaf = parts[-2] if len(parts) > 1 else parts[-1]
            acc[leaf].append(t.rel_change)
        return {k: sum(v) / len(v) for k, v in sorted(acc.items())}

    def summary(self) -> dict[str, Any]:
        changed = len(self.tensors)
        mean_rel = sum(t.rel_change for t in self.tensors) / changed if changed else 0.0
        top1 = [t.top1_energy for t in self.tensors if t.top1_energy is not None]
        return {
            "changed_tensors": changed,
            "unchanged_tensors": self.unchanged,
            "removed_tensors": len(self.only_in_base),
            "added_tensors": len(self.only_in_other),
            "shape_mismatches": len(self.shape_mismatch),
            "mean_rel_change": mean_rel,
            "max_rel_change": max((t.rel_change for t in self.tensors), default=0.0),
            "mean_top1_energy": sum(top1) / len(top1) if top1 else None,
            "layers_touched": len([k for k in self.by_layer() if k != "other"]),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "base": self.base,
            "other": self.other,
            "summary": self.summary(),
            "by_layer": self.by_layer(),
            "by_module": self.by_module(),
            "tensors": [asdict(t) for t in self.tensors],
            "only_in_base": self.only_in_base,
            "only_in_other": self.only_in_other,
            "shape_mismatch": self.shape_mismatch,
        }

    def to_markdown(self) -> str:
        s = self.summary()
        lines = [
            f"### Weight diff: `{self.base}` vs `{self.other}`",
            "",
            "| quantity | value |",
            "|---|---|",
        ]
        for k, v in s.items():
            lines.append(f"| {k} | {_fmt(v)} |")
        layers = self.by_layer()
        if layers:
            lines += ["", "| layer | max relative change |", "|---|---|"]
            lines += [f"| {k} | {v:.5f} |" for k, v in layers.items()]
        modules = self.by_module()
        if modules:
            lines += ["", "| module | mean relative change |", "|---|---|"]
            lines += [f"| {k} | {v:.5f} |" for k, v in modules.items()]
        return "\n".join(lines) + "\n"

    def save(self, out_dir: str | Path, stem: str = "diff") -> tuple[Path, Path]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        j, m = out / f"{stem}.json", out / f"{stem}.md"
        j.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        m.write_text(self.to_markdown(), encoding="utf-8")
        return j, m


def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def compare(a, b) -> tuple[float, float, float]:
    import torch

    a = a.float().flatten()
    b = b.float().flatten()
    delta = b - a
    na = a.norm().clamp_min(1e-12)
    rel = float(delta.norm() / na)
    cos = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
    tol = 1e-6 * float(a.abs().max().clamp_min(1e-12))
    frac = float((delta.abs() > tol).float().mean())
    return rel, cos, frac


def top1_energy(a, b) -> float | None:
    import torch

    if a.ndim != 2:
        return None
    d = (b.float() - a.float())
    s = torch.linalg.svdvals(d)
    total = float((s**2).sum())
    return float(s[0] ** 2 / total) if total > 0 else None


def diff(base: str | Path | TensorSource, other: str | Path | TensorSource, spectral: bool = False) -> DiffReport:
    src_a = base if isinstance(base, TensorSource) else TensorSource.open(base)
    src_b = other if isinstance(other, TensorSource) else TensorSource.open(other)
    is_patch = _is_patch(other)
    report = DiffReport(base=src_a.label, other=src_b.label)
    keys_b = src_b.keys()
    for key in keys_b:
        if key not in src_a:
            report.only_in_other.append(key)
            continue
        ta, tb = src_a.get(key), src_b.get(key)
        if tuple(ta.shape) != tuple(tb.shape):
            report.shape_mismatch.append(key)
            continue
        rel, cos, frac = compare(ta, tb)
        if frac == 0.0:
            report.unchanged += 1
            continue
        report.tensors.append(
            TensorDiff(
                name=key,
                shape=list(ta.shape),
                rel_change=rel,
                cosine=cos,
                frac_changed=frac,
                top1_energy=top1_energy(ta, tb) if spectral else None,
            )
        )
    if not is_patch:
        report.only_in_base = [k for k in src_a.keys() if k not in src_b]
    else:
        report.unchanged += len([k for k in src_a.keys() if k not in src_b])
    return report


def _is_patch(ref) -> bool:
    if isinstance(ref, TensorSource):
        return False
    p = Path(ref)
    return (p.is_dir() and (p / PATCH_META).exists()) or p.name == PATCH_WEIGHTS
