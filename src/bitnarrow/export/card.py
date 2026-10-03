from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from bitnarrow import __version__
from bitnarrow.gate import GateReport
from bitnarrow.recipe import Recipe
from bitnarrow.schema import RunResult

METHOD_BLURB = {
    "narrow": (
        "Training free directional weight edit. A behavior direction is isolated in the residual stream from "
        "contrastive prompt sets using Winsorized activation means, then projected out of the selected projection "
        "weights. Only the edited tensors ship, as a small patch that hot loads onto the stock base model."
    ),
    "prune": (
        "Depth pruning. Decoder blocks are ranked by the angular distance between their input and output "
        "residual streams on a calibration set, and the least influential contiguous block is removed."
    ),
    "merge": "Checkpoint merge produced with mergekit using the configuration recorded below.",
    "none": "Unmodified base model evaluated as a reference.",
}


def _fmt(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v:.4f}" if abs(v) < 1000 else f"{v:,.1f}"


def results_table(base: RunResult, artifact: RunResult) -> str:
    rows = ["| metric | category | base | artifact | delta | direction |", "|---|---|---|---|---|---|"]
    names = list(dict.fromkeys(list(base.metrics) + list(artifact.metrics)))
    for name in names:
        bm, am = base.metrics.get(name), artifact.metrics.get(name)
        ref = am or bm
        delta = (am.value - bm.value) if (am and bm) else None
        arrow = "higher is better" if ref.higher_is_better else "lower is better"
        rows.append(
            f"| {name} | {ref.category} | {_fmt(bm.value if bm else None)} | {_fmt(am.value if am else None)} | "
            f"{'n/a' if delta is None else f'{delta:+.4f}'} | {arrow} |"
        )
    return "\n".join(rows)


def gate_table(gate: GateReport) -> str:
    rows = ["| check | result | detail |", "|---|---|---|"]
    for c in gate.checks:
        rows.append(f"| {c.name} | {'pass' if c.passed else 'fail'} | {c.detail} |")
    return "\n".join(rows)


def sweep_table(sweep: list[dict[str, Any]]) -> str:
    if not sweep:
        return ""
    metric_names = list(dict.fromkeys(k for row in sweep for k in row.get("metrics", {})))
    head = "| strength | gate | " + " | ".join(metric_names) + " |"
    sep = "|---|---|" + "---|" * len(metric_names)
    rows = [head, sep]
    for row in sweep:
        vals = " | ".join(_fmt(row["metrics"].get(m)) for m in metric_names)
        rows.append(f"| {row['strength']} | {'pass' if row['passed'] else 'fail'} | {vals} |")
    return "\n".join(rows)


def usage(kind: str, repo_id: str, base_model: str) -> str:
    if kind == "patch":
        return f"""```bash
pip install "bitnarrow[torch]"
```

```python
import bitnarrow

model, tokenizer = bitnarrow.load("{repo_id}")
messages = [{{"role": "user", "content": "How do I kill a Python process that hangs?"}}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(inputs, max_new_tokens=128)[0][inputs.shape[1]:], skip_special_tokens=True))
```

4bit loading keeps the patched modules in full precision and quantizes the rest:

```python
model, tokenizer = bitnarrow.load("{repo_id}", load_in_4bit=True)
```

Materialize a standalone checkpoint (for vLLM, GGUF, or plain transformers):

```bash
bitnarrow export {repo_id} --materialize ./merged
vllm serve ./merged
```"""
    return f"""```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo_id}", device_map="auto", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
```

```bash
vllm serve {repo_id}
```"""


def render(
    recipe: Recipe,
    base: RunResult,
    artifact: RunResult,
    gate: GateReport,
    kind: str,
    repo_id: str | None = None,
    diff_summary: dict[str, Any] | None = None,
    sweep: list[dict[str, Any]] | None = None,
    method_meta: dict[str, Any] | None = None,
) -> str:
    repo_id = repo_id or recipe.output.repo_id or f"<your-namespace>/{recipe.name}"
    local_base = Path(recipe.base.model).exists()
    front = {
        "license": recipe.output.license or "other",
        "base_model": None if local_base else recipe.base.model,
        "library_name": "transformers" if kind == "model" else "bitnarrow",
        "pipeline_tag": "text-generation",
        "tags": list(dict.fromkeys(["bitnarrow", f"bitnarrow-{recipe.method}", *recipe.output.tags])),
    }
    front = {k: v for k, v in front.items() if v is not None}
    base_link = (
        f"`{Path(recipe.base.model).name}` (local)"
        if local_base
        else f"[`{recipe.base.model}`](https://huggingface.co/{recipe.base.model})"
    )
    status = "PASS" if gate.passed else "FAIL"
    meta = method_meta or {}
    meta_lines = "\n".join(f"| {k} | `{_short(v)}` |" for k, v in meta.items() if k not in {"layer_scores", "band_cosines"})
    parts = [
        "---",
        yaml.safe_dump(front, sort_keys=False).strip(),
        "---",
        "",
        f"# {repo_id.split('/')[-1]}",
        "",
        f"Produced by [bitnarrow](https://github.com/amareshhebbar/bitnarrow) v{__version__} with method "
        f"`{recipe.method}` on base model {base_link}.",
        "",
        f"**Release gate: {status}**",
        "",
        METHOD_BLURB.get(recipe.method, ""),
        "",
        "## Results",
        "",
        "Base and artifact were scored by the same harness, same prompts, same seed.",
        "",
        results_table(base, artifact),
        "",
        "## Release gate",
        "",
        "An artifact ships only if harmful prompt refusal is held at or above the base model and general "
        "capability stays within the configured tolerance.",
        "",
        gate_table(gate),
        "",
    ]
    if sweep:
        parts += ["## Strength sweep", "", sweep_table(sweep), ""]
    if meta_lines:
        parts += ["## Method parameters", "", "| parameter | value |", "|---|---|", meta_lines, ""]
    if diff_summary:
        parts += [
            "## Weight diff",
            "",
            "| quantity | value |",
            "|---|---|",
            *[f"| {k} | {_short(v)} |" for k, v in diff_summary.items()],
            "",
        ]
    parts += [
        "## Usage",
        "",
        usage(kind, repo_id, recipe.base.model),
        "",
        "## Reproduce",
        "",
        "```bash",
        'pip install "bitnarrow[all]"',
        f"bitnarrow run recipes/{recipe.method}/{recipe.name}.yaml",
        "```",
        "",
        "The full recipe, evaluation JSON, gate report, and diff report are in this repository.",
        "",
        "## Limitations",
        "",
        "Refusal is scored with a substring judge on the first tokens of greedy output, which can miss partial "
        "or deflecting refusals. Capability numbers come from limited sample counts recorded in "
        "`results/`. This artifact is released for interpretability and evaluation research.",
        "",
    ]
    return "\n".join(parts)


def _short(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    text = str(v)
    return text if len(text) <= 120 else text[:117] + "..."
