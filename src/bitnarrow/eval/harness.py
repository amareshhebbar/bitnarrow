from __future__ import annotations

import re
from pathlib import Path

from bitnarrow import env
from bitnarrow.eval import capability, refusal, systems
from bitnarrow.recipe import EvalSpec
from bitnarrow.schema import Metric, ModelRef, RunResult


def evaluate(model, tokenizer, spec: EvalSpec, ref: ModelRef, chat: bool = True, seed: int = 0) -> RunResult:
    env.seed_everything(seed)
    result = RunResult(model=ref, env=env.capture(seed), eval_config=_spec_dict(spec))
    for suite in spec.suites:
        if suite == "refusal" and spec.refusal:
            metrics, samples = refusal.run(model, tokenizer, spec.refusal, spec.batch_size, chat)
            result.extra["refusal_samples"] = samples
        elif suite == "capability" and spec.capability:
            metrics = capability.run(model, tokenizer, spec.capability, spec.batch_size)
        elif suite == "systems":
            metrics = systems.run(model, tokenizer, spec.systems, chat)
        else:
            continue
        for name, m in metrics.items():
            result.metrics[name] = Metric(**m)
    return result


def cache_path(cache_dir: str | Path, model_id: str, revision: str | None, spec: EvalSpec, seed: int) -> Path:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "__", f"{model_id}@{revision or 'main'}")
    return Path(cache_dir) / f"{slug}__{spec.fingerprint()}__s{seed}.json"


def evaluate_cached(model, tokenizer, spec: EvalSpec, ref: ModelRef, cache_dir, chat=True, seed=0, refresh=False):
    path = cache_path(cache_dir, ref.base, ref.revision, spec, seed)
    if path.exists() and not refresh:
        return RunResult.load(path), True
    result = evaluate(model, tokenizer, spec, ref, chat=chat, seed=seed)
    result.save(path)
    return result, False


def _spec_dict(spec: EvalSpec) -> dict:
    from dataclasses import asdict

    data = asdict(spec)
    data["fingerprint"] = spec.fingerprint()
    return data
