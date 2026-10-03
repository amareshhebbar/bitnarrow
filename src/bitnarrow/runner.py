from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bitnarrow import env
from bitnarrow.gate import GateReport
from bitnarrow.gate import evaluate as gate_evaluate
from bitnarrow.recipe import Recipe
from bitnarrow.schema import ModelRef, RunResult

Log = Callable[[str], None]


def _stdout(msg: str) -> None:
    print(f"[bitnarrow] {msg}", flush=True)


def run(
    recipe: Recipe,
    out_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    skip_eval: bool = False,
    refresh_base: bool = False,
    device_map: str | None = "auto",
    log: Log = _stdout,
) -> Path:
    from bitnarrow.eval.harness import evaluate, evaluate_cached
    from bitnarrow.models import load_model, load_tokenizer

    out = Path(out_dir or Path("runs") / recipe.name)
    if out.exists():
        shutil.rmtree(out / "artifact", ignore_errors=True)
    (out / "results").mkdir(parents=True, exist_ok=True)
    recipe.dump(out / "recipe.yaml")
    env.seed_everything(recipe.seed)
    cache = Path(cache_dir or Path("runs") / ".cache")
    chat = recipe.base.chat_template
    b = recipe.base

    log(f"loading {b.model} ({b.dtype})")
    tok = load_tokenizer(b.model, b.revision, b.trust_remote_code)
    model = load_model(b.model, b.revision, b.dtype, False, b.trust_remote_code, device_map=device_map)

    base_ref = ModelRef(base=b.model, revision=b.revision, label="base")
    base_result: RunResult | None = None
    if not skip_eval:
        log("scoring base model")
        base_result, hit = evaluate_cached(model, tok, recipe.eval, base_ref, cache, chat, recipe.seed, refresh_base)
        log(f"base scored{' (cache hit)' if hit else ''}: {_brief(base_result)}")
        base_result.save(out / "results" / "base.json")

    def score(m, label: str, patch: str | None = None, path: str | None = None) -> RunResult:
        ref = ModelRef(base=b.model, revision=b.revision, patch=patch, path=path, label=label)
        return evaluate(m, tok, recipe.eval, ref, chat=chat, seed=recipe.seed)

    manifest: dict[str, Any] = {
        "name": recipe.name,
        "method": recipe.method,
        "base_model": b.model,
        "base_revision": b.revision,
        "repo_id": recipe.output.repo_id,
        "private": recipe.output.private,
        "recipe_fingerprint": recipe.fingerprint(),
        "eval_skipped": skip_eval,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "env": env.capture(recipe.seed),
    }

    if recipe.method == "narrow":
        artifact_result, gate, extra = _run_narrow(recipe, model, tok, out, base_result, score, chat, skip_eval, log)
        manifest.update(extra)
        manifest["kind"] = "patch"
    elif recipe.method == "prune":
        artifact_result, gate, extra = _run_prune(recipe, model, tok, out, base_result, score, chat, skip_eval, log)
        manifest.update(extra)
        manifest["kind"] = "model"
    elif recipe.method == "merge":
        del model
        artifact_result, gate, extra = _run_merge(recipe, out, base_result, skip_eval, log, device_map)
        manifest.update(extra)
        manifest["kind"] = "model"
    else:
        artifact_result = base_result
        gate = gate_evaluate(base_result, base_result, recipe.gate) if base_result else None
        manifest["kind"] = "reference"

    if artifact_result is not None:
        artifact_result.save(out / "results" / "artifact.json")
    if gate is not None:
        (out / "gate.json").write_text(json.dumps(gate.to_dict(), indent=2), encoding="utf-8")
        log(gate.summary())
    manifest["gate_passed"] = bool(gate and gate.passed)
    manifest["run_id"] = artifact_result.run_id if artifact_result else None

    if manifest["kind"] in {"patch", "model"}:
        try:
            from bitnarrow.diff import diff

            log("computing weight diff")
            report = diff(b.model, out / "artifact")
            report.save(out)
            manifest["diff_summary"] = report.summary()
        except Exception as exc:
            log(f"diff skipped: {exc}")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    if base_result is not None and artifact_result is not None and gate is not None:
        render_card(out)
        log(f"model card written to {out / 'README.md'}")
    log(f"done: {out}")
    return out


def _run_narrow(recipe, model, tok, out, base_result, score, chat, skip_eval, log):
    from bitnarrow.models import apply_patch, write_patch
    from bitnarrow.surgery import narrow

    cfg = narrow.NarrowConfig.from_params(recipe.params)
    log("collecting activations and extracting direction")
    direction = narrow.find_direction(model, tok, cfg, chat)
    log(f"direction layer {direction.layer}, band {direction.band[0]}..{direction.band[-1]} ({len(direction.band)} layers)")
    strengths = cfg.strengths()
    sweep: list[dict[str, Any]] = []
    chosen: tuple[float, dict, RunResult | None, GateReport | None] | None = None
    originals: dict[str, Any] | None = None

    for s in (strengths if not skip_eval else strengths[-1:]):
        patch = narrow.build_patch(model, direction, cfg, s)
        if originals is None:
            originals = {k: model.get_parameter(k).detach().clone() for k in patch}
        if skip_eval:
            chosen = (s, patch, None, None)
            break
        apply_patch(model, patch)
        log(f"scoring strength {s}")
        res = score(model, f"strength={s}")
        res.save(out / "results" / f"sweep_strength_{s}.json")
        gate = gate_evaluate(base_result, res, recipe.gate)
        apply_patch(model, originals)
        sweep.append(
            {
                "strength": s,
                "passed": gate.passed,
                "metrics": {k: v.value for k, v in res.metrics.items() if v.category != "systems"},
            }
        )
        log(f"strength {s}: gate {'PASS' if gate.passed else 'FAIL'} {_brief(res)}")
        if chosen is None or _better(gate, s, chosen):
            chosen = (s, patch, res, gate)

    s, patch, res, gate = chosen
    meta = narrow.patch_meta(direction, cfg, s, recipe.base.model, recipe.base.revision)
    art = write_patch(patch, meta, out / "artifact")
    _save_direction(direction, art)
    if res is not None:
        res.model.patch = str(art)
        res.model.label = "artifact"
    return res, gate, {"chosen_strength": s, "sweep": sweep, "method_meta": _public_meta(meta)}


def _better(gate: GateReport, strength: float, chosen) -> bool:
    _, _, _, prev_gate = chosen
    if gate.passed != prev_gate.passed:
        return gate.passed
    if gate.passed:
        return strength > chosen[0]
    ok_now = sum(c.passed for c in gate.checks)
    ok_prev = sum(c.passed for c in prev_gate.checks)
    return ok_now > ok_prev or (ok_now == ok_prev and strength < chosen[0])


def _save_direction(direction, art: Path) -> None:
    from safetensors.torch import save_file

    save_file({"direction": direction.vector.contiguous()}, str(art / "direction.safetensors"))


def _public_meta(meta: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in meta.items() if k not in {"layer_scores", "band_cosines"}}
    if isinstance(out.get("band"), list) and out["band"]:
        out["band"] = f"{out['band'][0]}..{out['band'][-1]}"
    return out


def _run_prune(recipe, model, tok, out, base_result, score, chat, skip_eval, log):
    from bitnarrow.surgery import prune

    cfg = prune.PruneConfig.from_params(recipe.params)
    log("scoring block importance on calibration set")
    decision = prune.choose(model, tok, cfg, chat)
    log(f"removing layers {decision['remove']} of {decision['n_layers']}")
    kept = prune.remove_layers(model, decision["remove"])
    art = out / "artifact"
    art.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(art), safe_serialization=True)
    tok.save_pretrained(str(art))
    (art / "bitnarrow_prune.json").write_text(
        json.dumps({**decision, "base_model": recipe.base.model, "kept_layers": kept}, indent=2), encoding="utf-8"
    )
    res = gate = None
    if not skip_eval:
        res = score(model, "artifact", path=str(art))
        gate = gate_evaluate(base_result, res, recipe.gate)
    meta = {"mode": cfg.mode, "removed_layers": decision["remove"], "layers_before": decision["n_layers"], "layers_after": kept}
    return res, gate, {"method_meta": meta}


def _run_merge(recipe, out, base_result, skip_eval, log, device_map):
    from bitnarrow.eval.harness import evaluate
    from bitnarrow.models import load_model, load_tokenizer
    from bitnarrow.surgery.merge import run_mergekit

    if "mergekit" not in recipe.params:
        raise ValueError("merge params need a 'mergekit' configuration mapping")
    art = out / "artifact"
    log("running mergekit")
    run_mergekit(recipe.params["mergekit"], art, recipe.params.get("mergekit_args"))
    res = gate = None
    if not skip_eval:
        tok = load_tokenizer(str(art))
        model = load_model(str(art), dtype=recipe.base.dtype, device_map=device_map)
        ref = ModelRef(base=recipe.base.model, path=str(art), label="artifact")
        res = evaluate(model, tok, recipe.eval, ref, chat=recipe.base.chat_template, seed=recipe.seed)
        gate = gate_evaluate(base_result, res, recipe.gate)
    return res, gate, {"method_meta": {"merge_method": recipe.params["mergekit"].get("merge_method")}}


def render_card(run_dir: str | Path, repo_id: str | None = None) -> Path:
    from bitnarrow.export.card import render

    run = Path(run_dir)
    recipe = Recipe.load(run / "recipe.yaml")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    base = RunResult.load(run / "results" / "base.json")
    artifact = RunResult.load(run / "results" / "artifact.json")
    gate = GateReport.from_dict(json.loads((run / "gate.json").read_text(encoding="utf-8")))
    text = render(
        recipe,
        base,
        artifact,
        gate,
        kind=manifest.get("kind", "patch"),
        repo_id=repo_id or manifest.get("repo_id"),
        diff_summary=manifest.get("diff_summary"),
        sweep=manifest.get("sweep"),
        method_meta=manifest.get("method_meta"),
    )
    path = run / "README.md"
    path.write_text(text, encoding="utf-8")
    return path


def _brief(result: RunResult | None) -> str:
    if result is None:
        return ""
    keys = [k for k, m in result.metrics.items() if m.category != "systems"][:4]
    return " ".join(f"{k}={result.metrics[k].value:.4f}" for k in keys)
