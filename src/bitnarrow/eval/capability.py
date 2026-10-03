from __future__ import annotations

import math
from typing import Any

from bitnarrow.data import PromptSource
from bitnarrow.models import model_device

PREFERRED = [
    "acc_norm,none",
    "acc,none",
    "exact_match,strict-match",
    "exact_match,flexible-extract",
    "exact_match,none",
    "pass@1,none",
    "inst_level_strict_acc,none",
]


def perplexity(model, tokenizer, texts: list[str], max_length: int = 512, batch_size: int = 4) -> float:
    import torch

    device = model_device(model)
    total_nll, total_tokens = 0.0, 0
    side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(
                texts[i : i + batch_size], return_tensors="pt", padding=True, truncation=True, max_length=max_length
            )
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            with torch.no_grad():
                logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits.float()
            shift_logits = logits[:, :-1]
            shift_labels = ids[:, 1:]
            shift_mask = mask[:, 1:].bool()
            nll = torch.nn.functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]), shift_labels.reshape(-1), reduction="none"
            ).view_as(shift_labels)
            total_nll += float(nll[shift_mask].sum())
            total_tokens += int(shift_mask.sum())
    finally:
        tokenizer.padding_side = side
    if total_tokens == 0:
        raise ValueError("perplexity texts produced zero tokens")
    return math.exp(total_nll / total_tokens)


def _pick(task_metrics: dict[str, Any]) -> tuple[str, float, float | None] | None:
    for key in PREFERRED:
        if key in task_metrics and isinstance(task_metrics[key], (int, float)):
            stderr_key = key.replace(",", "_stderr,", 1)
            stderr = task_metrics.get(stderr_key)
            return key, float(task_metrics[key]), float(stderr) if isinstance(stderr, (int, float)) else None
    for key, value in task_metrics.items():
        if isinstance(value, (int, float)) and "stderr" not in key and key != "alias":
            return key, float(value), None
    return None


def lm_eval_tasks(model, tokenizer, spec: dict[str, Any], batch_size: int) -> dict[str, dict[str, Any]]:
    try:
        from lm_eval import simple_evaluate
        from lm_eval.models.huggingface import HFLM
    except ImportError as exc:
        raise ImportError("install bitnarrow[eval] for lm-evaluation-harness tasks") from exc
    side = tokenizer.padding_side
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=spec.get("batch_size", batch_size))
    try:
        out = simple_evaluate(
            model=lm,
            tasks=list(spec["tasks"]),
            num_fewshot=spec.get("num_fewshot"),
            limit=spec.get("limit"),
            apply_chat_template=bool(spec.get("apply_chat_template", False)),
            random_seed=int(spec.get("seed", 0)),
            numpy_random_seed=int(spec.get("seed", 0)),
            torch_random_seed=int(spec.get("seed", 0)),
        )
    finally:
        tokenizer.padding_side = side
    metrics: dict[str, dict[str, Any]] = {}
    for task, values in (out or {}).get("results", {}).items():
        picked = _pick(values)
        if picked is None:
            continue
        key, value, stderr = picked
        metrics[task] = {
            "value": value,
            "category": "capability",
            "higher_is_better": True,
            "stderr": stderr,
        }
    return metrics


def run(model, tokenizer, spec: dict[str, Any], batch_size: int = 8) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    if spec.get("tasks"):
        metrics.update(lm_eval_tasks(model, tokenizer, spec, batch_size))
    if spec.get("perplexity"):
        pspec = dict(spec["perplexity"])
        max_length = int(pspec.pop("max_length", 512))
        texts = PromptSource.from_dict(pspec).load()
        ppl = perplexity(model, tokenizer, texts, max_length=max_length, batch_size=max(1, batch_size // 2))
        metrics["perplexity"] = {
            "value": ppl,
            "category": "capability",
            "higher_is_better": False,
            "n": len(texts),
        }
    return metrics
