from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

PUBLISH_PATTERNS = [
    "README.md",
    "recipe.yaml",
    "gate.json",
    "diff.json",
    "diff.md",
    "results/*.json",
    "artifact/*",
]


def publish(run_dir: str | Path, repo_id: str | None = None, private: bool | None = None, force: bool = False) -> str:
    from huggingface_hub import HfApi

    run = Path(run_dir)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    repo_id = repo_id or manifest.get("repo_id")
    if not repo_id:
        raise ValueError("no repo_id given and none recorded in the recipe output section")
    if not manifest.get("gate_passed") and not force:
        raise RuntimeError("gate failed for this run; refusing to publish (use --force to override deliberately)")
    private = manifest.get("private", False) if private is None else private
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    staging = run / "_publish"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    for pattern in PUBLISH_PATTERNS:
        for src in run.glob(pattern):
            if not src.is_file():
                continue
            rel = src.relative_to(run)
            if rel.parts[0] == "artifact":
                rel = Path(*rel.parts[1:])
            dst = staging / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(src, dst)
            except OSError:
                shutil.copyfile(src, dst)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(staging),
        commit_message=f"bitnarrow publish {manifest.get('name')} ({manifest.get('run_id')})",
    )
    shutil.rmtree(staging, ignore_errors=True)
    return f"https://huggingface.co/{repo_id}"
