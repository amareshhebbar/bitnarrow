from __future__ import annotations

import json
from pathlib import Path

import pytest

from bitnarrow.data import PromptSource
from bitnarrow.eval.refusal import is_refusal
from bitnarrow.gate import GateConfig, GateReport, evaluate
from bitnarrow.recipe import Recipe
from bitnarrow.schema import ModelRef, RunResult

ROOT = Path(__file__).resolve().parents[1]


def _result(**metrics) -> RunResult:
    r = RunResult(model=ModelRef(base="m"))
    for name, (value, category, hib) in metrics.items():
        r.add(name, value, category, hib)
    return r


@pytest.mark.parametrize("path", sorted((ROOT / "recipes").rglob("*.yaml")), ids=lambda p: p.stem)
def test_shipped_recipes_parse(path):
    recipe = Recipe.load(path)
    assert recipe.name == path.stem
    assert recipe.fingerprint() == Recipe.from_dict(recipe.to_dict()).fingerprint()


def test_recipe_rejects_bad_input():
    with pytest.raises(ValueError):
        Recipe.from_dict({"name": "x", "method": "nope", "base": "m"})
    with pytest.raises(ValueError):
        Recipe.from_dict({"name": "x", "method": "narrow", "base": "m", "surprise": 1})
    with pytest.raises(ValueError):
        Recipe.from_dict({"name": "a/b", "method": "narrow", "base": "m"})
    with pytest.raises(ValueError):
        Recipe.from_dict({"name": "x", "method": "narrow", "base": "m", "eval": {"suites": ["vibes"]}})


def test_recipe_dump_roundtrip(tmp_path):
    r = Recipe.from_dict({"name": "x", "method": "prune", "base": "m", "gate": {"require": [{"metric": "a"}]}})
    p = r.dump(tmp_path / "r.yaml")
    assert Recipe.load(p).to_dict() == r.to_dict()


def test_schema_roundtrip(tmp_path):
    r = _result(acc=(0.5, "capability", True))
    loaded = RunResult.load(r.save(tmp_path / "r.json"))
    assert loaded.metrics["acc"].value == 0.5
    assert loaded.run_id == r.run_id
    data = json.loads((tmp_path / "r.json").read_text())
    data["schema_version"] = "9.0"
    with pytest.raises(ValueError):
        RunResult.from_dict(data)
    with pytest.raises(ValueError):
        r.add("bad", 1.0, "vibes")


def test_gate_passes_clean_edit():
    base = _result(harm=(0.95, "harmful_refusal", True), acc=(0.60, "capability", True), orr=(0.40, "over_refusal", False))
    art = _result(harm=(0.96, "harmful_refusal", True), acc=(0.59, "capability", True), orr=(0.10, "over_refusal", False))
    cfg = GateConfig.from_dict({"max_capability_drop": 0.02, "require": [{"metric": "orr", "direction": "decrease", "min_delta": 0.05}]})
    report = evaluate(base, art, cfg)
    assert report.passed, report.summary()
    assert GateReport.from_dict(report.to_dict()).passed


def test_gate_blocks_safety_regression():
    base = _result(harm=(0.95, "harmful_refusal", True), acc=(0.60, "capability", True))
    art = _result(harm=(0.50, "harmful_refusal", True), acc=(0.70, "capability", True))
    report = evaluate(base, art, GateConfig())
    assert not report.passed
    assert any(c.name.startswith("harmful_refusal") and not c.passed for c in report.checks)


def test_gate_capability_lower_is_better_and_overrides():
    base = _result(ppl=(10.0, "capability", False))
    worse = _result(ppl=(10.4, "capability", False))
    assert not evaluate(base, worse, GateConfig()).passed
    assert evaluate(base, worse, GateConfig.from_dict({"overrides": {"ppl": 0.5}})).passed
    assert evaluate(base, worse, GateConfig.from_dict({"capability_drop_mode": "relative", "max_capability_drop": 0.05})).passed


def test_gate_missing_metric_and_empty():
    base = _result(acc=(0.6, "capability", True))
    assert not evaluate(base, _result(), GateConfig()).passed
    assert not evaluate(_result(), _result(), GateConfig()).passed


def test_gate_require_increase():
    base = _result(acc=(0.6, "capability", True))
    art = _result(acc=(0.65, "capability", True))
    cfg = GateConfig.from_dict({"require": [{"metric": "acc", "direction": "increase", "min_delta": 0.1}]})
    assert not evaluate(base, art, cfg).passed


@pytest.mark.parametrize(
    "text,expected",
    [
        ("I'm sorry, but I can't help with that.", True),
        ("I’m sorry, I cannot do that", True),
        ("As an AI language model, I do not", True),
        ("", True),
        ("Sure. To kill a hung Python process, run kill -9 with its PID.", False),
        ("Here are three steps to terminate the thread.", False),
    ],
)
def test_refusal_judge(text, expected):
    assert is_refusal(text) is expected


def test_prompt_sources(tmp_path):
    assert PromptSource.from_dict(["a", "b"]).load() == ["a", "b"]
    (tmp_path / "p.txt").write_text("one\n\ntwo\n")
    assert PromptSource.from_dict(str(tmp_path / "p.txt")).load() == ["one", "two"]
    rows = [{"prompt": f"q{i}", "label": "safe" if i % 2 else "unsafe"} for i in range(10)]
    (tmp_path / "p.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    src = PromptSource.from_dict({"file": str(tmp_path / "p.jsonl"), "filter": {"label": "safe"}, "limit": 3, "seed": 1})
    first, second = src.load(), src.load()
    assert first == second and len(first) == 3
    assert all(int(p[1:]) % 2 == 1 for p in first)
    with pytest.raises(ValueError):
        PromptSource.from_dict({"file": "a.txt", "prompts": ["x"]})
    with pytest.raises(ValueError):
        PromptSource.from_dict({"bogus": 1})


def test_card_render():
    from bitnarrow.export.card import render

    recipe = Recipe.load(ROOT / "recipes" / "narrow" / "qwen2.5-1.5b-narrow.yaml")
    base = _result(harmful_refusal_rate=(0.95, "harmful_refusal", True), arc_easy=(0.7, "capability", True),
                   over_refusal_rate=(0.3, "over_refusal", False))
    art = _result(harmful_refusal_rate=(0.95, "harmful_refusal", True), arc_easy=(0.695, "capability", True),
                  over_refusal_rate=(0.1, "over_refusal", False))
    gate = evaluate(base, art, recipe.gate)
    text = render(recipe, base, art, gate, kind="patch", sweep=[{"strength": 1.0, "passed": True, "metrics": {"arc_easy": 0.695}}],
                  method_meta={"strength": 1.0, "band": "0..27"}, diff_summary={"changed_tensors": 112})
    assert text.startswith("---\nlicense: apache-2.0")
    assert "base_model: Qwen/Qwen2.5-1.5B-Instruct" in text
    assert "Release gate: PASS" in text
    assert "| over_refusal_rate | over_refusal | 0.3000 | 0.1000 | -0.2000 | lower is better |" in text
    assert 'bitnarrow.load("AmareshHebbar/Qwen2.5-1.5B-Narrow")' in text


def test_cli_init_and_gate(tmp_path, capsys):
    from bitnarrow.cli import main

    out = tmp_path / "r.yaml"
    assert main(["init", "narrow", "demo", "--base", "Qwen/Qwen2.5-0.5B-Instruct", "--out", str(out)]) == 0
    recipe = Recipe.load(out)
    assert recipe.output.repo_id == "AmareshHebbar/Qwen2.5-0.5B-Instruct-Narrow"
    b = _result(acc=(0.6, "capability", True)).save(tmp_path / "b.json")
    a = _result(acc=(0.5, "capability", True)).save(tmp_path / "a.json")
    assert main(["gate", str(b), str(a)]) == 2
    assert main(["gate", str(b), str(b)]) == 0
