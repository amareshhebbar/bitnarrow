from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PromptSource:
    dataset: str | None = None
    file: str | None = None
    prompts: list[str] | None = None
    config: str | None = None
    split: str = "train"
    column: str = "prompt"
    filter: dict[str, Any] = field(default_factory=dict)
    limit: int | None = None
    seed: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | list[str] | str) -> PromptSource:
        if isinstance(data, list):
            return cls(prompts=[str(p) for p in data])
        if isinstance(data, str):
            return cls(file=data) if Path(data).suffix in {".txt", ".jsonl", ".json"} else cls(dataset=data)
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown prompt source keys: {sorted(unknown)}")
        source = cls(**data)
        if sum(x is not None for x in (source.dataset, source.file, source.prompts)) != 1:
            raise ValueError("prompt source needs exactly one of dataset, file, prompts")
        return source

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v not in (None, {}, [])}

    def load(self) -> list[str]:
        if self.prompts is not None:
            rows = [{self.column: p} for p in self.prompts]
        elif self.file is not None:
            rows = _read_file(Path(self.file), self.column)
        else:
            rows = _read_dataset(self.dataset, self.config, self.split)
        rows = [r for r in rows if all(r.get(k) == v for k, v in self.filter.items())]
        prompts = [str(r[self.column]) for r in rows if r.get(self.column)]
        if self.limit is not None and len(prompts) > self.limit:
            rng = random.Random(self.seed)
            prompts = rng.sample(prompts, self.limit)
        return prompts


def _read_file(path: Path, column: str) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".txt":
        return [{column: line.strip()} for line in text.splitlines() if line.strip()]
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, list):
        return [d if isinstance(d, dict) else {column: d} for d in data]
    raise ValueError(f"unsupported json layout in {path}")


def _read_dataset(name: str, config: str | None, split: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("install bitnarrow[eval] to load HuggingFace datasets") from exc
    ds = load_dataset(name, config, split=split)
    return [dict(row) for row in ds]
