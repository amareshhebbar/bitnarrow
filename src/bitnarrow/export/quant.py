from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from bitnarrow.models import load


def materialize(ref: str, out_dir: str | Path, patch: str | None = None, dtype: str = "bfloat16") -> Path:
    model, tok = load(ref, patch=patch, dtype=dtype)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out), safe_serialization=True)
    tok.save_pretrained(str(out))
    return out


def _llama_cpp(llama_cpp_dir: str | None) -> Path:
    root = Path(llama_cpp_dir or os.environ.get("LLAMA_CPP_DIR", ""))
    if not root or not (root / "convert_hf_to_gguf.py").exists():
        raise RuntimeError("set LLAMA_CPP_DIR or pass --llama-cpp to a llama.cpp checkout")
    return root


def to_gguf(model_dir: str | Path, out_dir: str | Path, quants: list[str], llama_cpp_dir: str | None = None) -> list[Path]:
    root = _llama_cpp(llama_cpp_dir)
    model_dir, out = Path(model_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = model_dir.name
    f16 = out / f"{stem}-F16.gguf"
    subprocess.run(
        [sys.executable, str(root / "convert_hf_to_gguf.py"), str(model_dir), "--outfile", str(f16), "--outtype", "f16"],
        check=True,
    )
    quantize = shutil.which("llama-quantize") or str(root / "build" / "bin" / "llama-quantize")
    produced = [f16]
    for q in quants:
        if q.upper() in {"F16", "FP16"}:
            continue
        target = out / f"{stem}-{q.upper()}.gguf"
        subprocess.run([quantize, str(f16), str(target), q.upper()], check=True)
        produced.append(target)
    return produced
