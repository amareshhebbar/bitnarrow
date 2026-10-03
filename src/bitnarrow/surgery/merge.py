from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml


def run_mergekit(config: dict[str, Any], out_dir: str | Path, extra_args: list[str] | None = None) -> Path:
    exe = shutil.which("mergekit-yaml")
    if exe is None:
        raise RuntimeError("mergekit-yaml not found; install bitnarrow[merge]")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = out.parent / "mergekit.yaml"
    cfg_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    cmd = [exe, str(cfg_path), str(out), "--copy-tokenizer", *(extra_args or [])]
    subprocess.run(cmd, check=True)
    return out
