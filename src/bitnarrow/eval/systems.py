from __future__ import annotations

import time
from typing import Any

from bitnarrow.models import format_prompts, model_device

PROMPT = "Explain in a short paragraph how a transformer language model generates text."


def run(model, tokenizer, spec: dict[str, Any], chat: bool = True) -> dict[str, dict[str, Any]]:
    import torch

    new_tokens = int(spec.get("new_tokens", 64))
    repeats = int(spec.get("repeats", 3))
    device = model_device(model)
    params = sum(p.numel() for p in model.parameters())
    size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    text = format_prompts(tokenizer, [spec.get("prompt", PROMPT)], chat)[0]
    batch = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
    gen = {
        "max_new_tokens": new_tokens,
        "min_new_tokens": new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
    }
    cuda = device.type == "cuda"
    with torch.no_grad():
        model.generate(**batch, **{**gen, "max_new_tokens": 4, "min_new_tokens": 4})
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            model.generate(**batch, **gen)
            if cuda:
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
    best = min(times)
    metrics = {
        "params_b": {"value": params / 1e9, "category": "systems", "higher_is_better": False},
        "weights_gb": {"value": size_bytes / 1024**3, "category": "systems", "higher_is_better": False},
        "tokens_per_second": {"value": new_tokens / best, "category": "systems", "higher_is_better": True},
    }
    if cuda:
        metrics["peak_vram_gb"] = {
            "value": torch.cuda.max_memory_allocated() / 1024**3,
            "category": "systems",
            "higher_is_better": False,
        }
    return metrics
