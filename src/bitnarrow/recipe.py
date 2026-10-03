from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from bitnarrow.gate import GateConfig

METHODS = {"narrow", "prune", "merge", "none"}
SUITES = {"refusal", "capability", "systems"}


@dataclass
class BaseSpec:
    model: str
    revision: str | None = None
    dtype: str = "bfloat16"
    load_in_4bit: bool = False
    trust_remote_code: bool = False
    chat_template: bool = True


@dataclass
class EvalSpec:
    suites: list[str] = field(default_factory=lambda: ["refusal", "capability", "systems"])
    refusal: dict[str, Any] = field(default_factory=dict)
    capability: dict[str, Any] = field(default_factory=dict)
    systems: dict[str, Any] = field(default_factory=dict)
    batch_size: int = 8

    def __post_init__(self) -> None:
        bad = set(self.suites) - SUITES
        if bad:
            raise ValueError(f"unknown eval suites {sorted(bad)}; allowed {sorted(SUITES)}")

    def fingerprint(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


@dataclass
class OutputSpec:
    repo_id: str | None = None
    license: str | None = None
    private: bool = False
    tags: list[str] = field(default_factory=list)


@dataclass
class Recipe:
    name: str
    method: str
    base: BaseSpec
    params: dict[str, Any] = field(default_factory=dict)
    eval: EvalSpec = field(default_factory=EvalSpec)
    gate: GateConfig = field(default_factory=GateConfig)
    output: OutputSpec = field(default_factory=OutputSpec)
    seed: int = 0
    source_path: str | None = None

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"unknown method {self.method!r}; allowed {sorted(METHODS)}")
        if not self.name or any(c in self.name for c in "/\\ "):
            raise ValueError("recipe name must be non empty without slashes or spaces")

    @classmethod
    def from_dict(cls, data: dict[str, Any], source_path: str | None = None) -> Recipe:
        data = dict(data)
        allowed = {"name", "method", "base", "params", "eval", "gate", "output", "seed"}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"unknown recipe keys {sorted(unknown)}")
        for key in ("name", "method", "base"):
            if key not in data:
                raise ValueError(f"recipe missing required key {key!r}")
        base = data["base"]
        if isinstance(base, str):
            base = {"model": base}
        return cls(
            name=data["name"],
            method=data["method"],
            base=BaseSpec(**base),
            params=dict(data.get("params") or {}),
            eval=EvalSpec(**(data.get("eval") or {})),
            gate=GateConfig.from_dict(data.get("gate")),
            output=OutputSpec(**(data.get("output") or {})),
            seed=int(data.get("seed", 0)),
            source_path=source_path,
        )

    @classmethod
    def load(cls, path: str | Path) -> Recipe:
        path = Path(path)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{path} is not a recipe mapping")
        return cls.from_dict(data, source_path=str(path))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "method": self.method,
            "base": asdict(self.base),
            "params": self.params,
            "eval": asdict(self.eval),
            "gate": self.gate.to_dict(),
            "output": asdict(self.output),
            "seed": self.seed,
        }

    def dump(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
        return path

    def fingerprint(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:12]
