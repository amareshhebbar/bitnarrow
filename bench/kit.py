from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

KIT_VERSION = "1.0.0"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_env(path: str | Path = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), value)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(canonical(r) + "\n")
            n += 1
    return n


def append_jsonl(path: str | Path, row: dict, lock: threading.Lock | None = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = canonical(row) + "\n"
    if lock:
        with lock:
            with open(p, "a", encoding="utf-8") as f:
                f.write(line)
    else:
        with open(p, "a", encoding="utf-8") as f:
            f.write(line)


def read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def merkle_root(leaves: list[str]) -> str:
    if not leaves:
        return sha256_text("")
    level = [bytes.fromhex(x) for x in leaves]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [hashlib.sha256(level[i] + level[i + 1]).digest() for i in range(0, len(level), 2)]
    return level[0].hex()


def jsonl_merkle(path: str | Path) -> tuple[str, int]:
    lines = [x for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    return merkle_root([sha256_text(x) for x in lines]), len(lines)


SKIP_PARTS = {"private", "__pycache__", ".cache", "_controls"}


def build_manifest(root: str | Path, folders: list[str], extra: dict | None = None) -> dict:
    root = Path(root)
    files = []
    for folder in folders:
        base = root / folder
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file() or SKIP_PARTS & set(p.relative_to(root).parts):
                continue
            rel = p.relative_to(root).as_posix()
            entry = {"path": rel, "sha256": sha256_file(p), "bytes": p.stat().st_size}
            if p.suffix == ".jsonl":
                entry["rows_merkle"], entry["rows"] = jsonl_merkle(p)
            files.append(entry)
    manifest = {
        "kit_version": KIT_VERSION,
        "created_at": now(),
        "files": files,
        "root": merkle_root([f["sha256"] for f in files]),
    }
    if extra:
        manifest.update(extra)
    write_json(root / "manifest.json", manifest)
    return manifest


def check_manifest(root: str | Path) -> list[str]:
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    problems = []
    for f in manifest["files"]:
        p = root / f["path"]
        if not p.exists():
            problems.append(f"missing {f['path']}")
            continue
        if sha256_file(p) != f["sha256"]:
            problems.append(f"hash mismatch {f['path']}")
    if merkle_root([f["sha256"] for f in manifest["files"]]) != manifest["root"]:
        problems.append("root mismatch")
    return problems


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return out.stdout.strip() or None
    except Exception:
        return None


def env_capture(packages: list[str] | None = None, repo: str | Path = ".") -> dict:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "argv": sys.argv,
        "git_commit": _run(["git", "-C", str(repo), "rev-parse", "HEAD"]),
        "git_dirty": bool(_run(["git", "-C", str(repo), "status", "--porcelain"])),
        "gpu": _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
    }
    versions = {}
    for mod in packages or []:
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = None
    info["packages"] = versions
    freeze = _run([sys.executable, "-m", "pip", "freeze"])
    info["pip_freeze_sha256"] = sha256_text(freeze or "")
    info["pip_freeze"] = (freeze or "").splitlines()
    return info


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - m) / d), min(1.0, (c + m) / d))


def paired_bootstrap(a: list[float], b: list[float], resamples: int = 10000, seed: int = 0) -> dict:
    if len(a) != len(b):
        raise ValueError("paired bootstrap needs equal length lists")
    n = len(a)
    if n == 0:
        return {"n": 0, "a": None, "b": None, "delta": None, "ci_low": None, "ci_high": None}
    import numpy as np

    diffs = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    rng = np.random.default_rng(seed)
    stats = np.empty(resamples, dtype=np.float64)
    chunk = max(1, min(resamples, 2_000_000 // max(1, n)))
    for s in range(0, resamples, chunk):
        k = min(chunk, resamples - s)
        idx = rng.integers(0, n, size=(k, n))
        stats[s : s + k] = diffs[idx].mean(axis=1)
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return {
        "n": n,
        "a": float(np.mean(a)),
        "b": float(np.mean(b)),
        "delta": float(diffs.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
    }


def mcnemar_exact(a: list[int], b: list[int]) -> float:
    n01 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n10 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def cohen_kappa(x: list, y: list) -> float | None:
    if len(x) != len(y) or not x:
        return None
    labels = sorted(set(x) | set(y), key=str)
    n = len(x)
    po = sum(1 for a, b in zip(x, y) if a == b) / n
    pe = sum((x.count(lab) / n) * (y.count(lab) / n) for lab in labels)
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def pct(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None else f"{100 * x:.{digits}f}%"


def pp(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None else f"{100 * x:+.{digits}f} pts"


class Budget:
    def __init__(self, max_usd: float, price_in: float, price_out: float):
        self.max_usd = max_usd
        self.price_in = price_in
        self.price_out = price_out
        self.tokens_in = 0
        self.tokens_out = 0
        self.lock = threading.Lock()

    @property
    def usd(self) -> float:
        return self.tokens_in / 1e6 * self.price_in + self.tokens_out / 1e6 * self.price_out

    def add(self, tin: int, tout: int) -> None:
        with self.lock:
            self.tokens_in += tin
            self.tokens_out += tout

    def exceeded(self) -> bool:
        return self.max_usd > 0 and self.usd >= self.max_usd


class ChatClient:
    def __init__(self, base_url: str, api_key: str, model: str, rpm: float = 0, budget: Budget | None = None):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key, max_retries=0, timeout=120)
        self.model = model
        self.interval = 60.0 / rpm if rpm and rpm > 0 else 0.0
        self.next_at = 0.0
        self.lock = threading.Lock()
        self.budget = budget

    def _wait(self) -> None:
        if not self.interval:
            return
        with self.lock:
            t = time.monotonic()
            delay = max(0.0, self.next_at - t)
            self.next_at = max(t, self.next_at) + self.interval
        if delay:
            time.sleep(delay)

    def chat(self, system: str, user: str, max_tokens: int = 16, temperature: float = 0.0, tries: int = 8,
             extra: dict | None = None) -> str:
        last = None
        for attempt in range(tries):
            if self.budget and self.budget.exceeded():
                raise RuntimeError(f"budget of ${self.budget.max_usd:.2f} reached")
            self._wait()
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    extra_body=extra or None,
                )
                if self.budget and resp.usage:
                    self.budget.add(resp.usage.prompt_tokens or 0, resp.usage.completion_tokens or 0)
                return (resp.choices[0].message.content or "").strip()
            except Exception as exc:
                last = exc
                text = str(exc).lower()
                if "budget" in text:
                    raise
                wait = min(90.0, 2.0 * (2**attempt)) + random.random()
                if "429" in text or "rate" in text:
                    wait = max(wait, 20.0)
                time.sleep(wait)
        raise RuntimeError(f"chat failed after {tries} tries: {last}")


def hf_upload(folder: str | Path, repo_id: str, private: bool = False, message: str = "bench results") -> str:
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    info = api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=message,
        ignore_patterns=["**/__pycache__/**", "**/.cache/**", "**/private/**", "private/**", "**/_controls/**", "*.safetensors", "**/*.safetensors"],
    )
    return getattr(info, "oid", None) or str(info)


def headline_url(repo_id: str, path: str = "report/HEADLINE.md") -> str:
    return f"https://huggingface.co/datasets/{repo_id}/blob/main/{path}"
