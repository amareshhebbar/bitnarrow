from __future__ import annotations

import platform
import random
import subprocess
from typing import Any

from bitnarrow import __version__


def capture(seed: int | None = None) -> dict[str, Any]:
    info: dict[str, Any] = {
        "bitnarrow": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_sha": _git_sha(),
        "seed": seed,
    }
    for mod in ("torch", "transformers", "safetensors", "datasets", "lm_eval", "bitsandbytes"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    try:
        import torch

        if torch.cuda.is_available():
            info["device"] = torch.cuda.get_device_name(0)
            info["device_count"] = torch.cuda.device_count()
            info["cuda"] = torch.version.cuda
        else:
            info["device"] = "cpu"
    except ImportError:
        info["device"] = None
    return info


def seed_everything(seed: int) -> None:
    random.seed(seed)
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


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5, check=True
        )
        return out.stdout.strip() or None
    except Exception:
        return None
