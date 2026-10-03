from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"

CATEGORIES = {"capability", "harmful_refusal", "over_refusal", "systems", "custom"}


@dataclass
class Metric:
    value: float
    category: str
    higher_is_better: bool = True
    n: int | None = None
    stderr: float | None = None

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES:
            raise ValueError(f"unknown metric category {self.category!r}")
        self.value = float(self.value)


@dataclass
class ModelRef:
    base: str
    revision: str | None = None
    patch: str | None = None
    path: str | None = None
    label: str = "base"


@dataclass
class RunResult:
    model: ModelRef
    metrics: dict[str, Metric] = field(default_factory=dict)
    env: dict[str, Any] = field(default_factory=dict)
    eval_config: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    schema_version: str = SCHEMA_VERSION

    def add(self, name: str, value: float, category: str, higher_is_better: bool = True, **kw: Any) -> None:
        self.metrics[name] = Metric(value=value, category=category, higher_is_better=higher_is_better, **kw)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["metrics"] = {k: asdict(v) for k, v in self.metrics.items()}
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunResult:
        major = str(data.get("schema_version", "0")).split(".")[0]
        if major != SCHEMA_VERSION.split(".")[0]:
            raise ValueError(f"incompatible schema_version {data.get('schema_version')}")
        data = dict(data)
        data["model"] = ModelRef(**data["model"])
        data["metrics"] = {k: Metric(**v) for k, v in data.get("metrics", {}).items()}
        return cls(**data)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> RunResult:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
