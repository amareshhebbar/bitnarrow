from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import bitnarrow
from bitnarrow.cli import main
from bitnarrow.diff import diff
from bitnarrow.runner import render_card, run
from conftest import tiny_recipe

pytestmark = pytest.mark.torch


def _quiet(_msg: str) -> None:
    return None


def test_narrow_run_end_to_end(tiny_model_dir, tmp_path):
    recipe = tiny_recipe(tiny_model_dir, "narrow")
    out = run(recipe, tmp_path / "run", tmp_path / "cache", log=_quiet)
    for rel in ("recipe.yaml", "manifest.json", "gate.json", "README.md", "diff.json", "diff.md",
                "results/base.json", "results/artifact.json", "results/sweep_strength_0.5.json",
                "artifact/patch.safetensors", "artifact/patch.json", "artifact/direction.safetensors"):
        assert (out / rel).exists(), rel
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["kind"] == "patch"
    assert manifest["gate_passed"] is True
    assert manifest["chosen_strength"] == 1.0
    assert [row["strength"] for row in manifest["sweep"]] == [0.5, 1.0]
    assert manifest["diff_summary"]["changed_tensors"] > 0
    assert manifest["diff_summary"]["removed_tensors"] == 0
    card = (out / "README.md").read_text()
    assert "Release gate: PASS" in card and "## Strength sweep" in card
    assert len(list((tmp_path / "cache").glob("*.json"))) == 1

    model, tok = bitnarrow.load(str(out / "artifact"), dtype="float32")
    patch = bitnarrow.models.read_patch(out / "artifact")[0]
    name = next(iter(patch))
    assert torch.equal(model.get_parameter(name).cpu(), patch[name])

    rerun = run(recipe, tmp_path / "run2", tmp_path / "cache", log=_quiet)
    assert json.loads((rerun / "results" / "base.json").read_text())["run_id"] == json.loads(
        (out / "results" / "base.json").read_text()
    )["run_id"]

    assert render_card(out, repo_id="me/renamed").read_text().count("me/renamed") >= 2
    assert main(["gate", str(out / "results/base.json"), str(out / "results/artifact.json"), "--recipe", str(out / "recipe.yaml")]) == 0
    assert main(["diff", str(tiny_model_dir), str(out / "artifact"), "--spectral", "--out", str(tmp_path / "d")]) == 0
    report = json.loads((tmp_path / "d" / "diff.json").read_text())
    energies = [t["top1_energy"] for t in report["tensors"]]
    assert all(e is not None and e > 0.99 for e in energies)


def test_narrow_strict_gate_failure(tiny_model_dir, tmp_path):
    recipe = tiny_recipe(
        tiny_model_dir,
        "narrow",
        gate={"harmful_refusal_tolerance": 1.0, "max_capability_drop": 1e9,
              "require": [{"metric": "perplexity", "direction": "decrease", "min_delta": 1e9}]},
    )
    out = run(recipe, tmp_path / "run", tmp_path / "cache", log=_quiet)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["gate_passed"] is False
    assert "Release gate: FAIL" in (out / "README.md").read_text()
    from bitnarrow.hub import publish

    with pytest.raises(RuntimeError):
        publish(out)


def test_prune_run_end_to_end(tiny_model_dir, tmp_path):
    recipe = tiny_recipe(tiny_model_dir, "prune")
    out = run(recipe, tmp_path / "run", tmp_path / "cache", log=_quiet)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["kind"] == "model"
    assert manifest["method_meta"]["layers_after"] == 4
    assert (out / "artifact" / "config.json").exists()
    model, _ = bitnarrow.load(str(out / "artifact"), dtype="float32")
    assert len(model.model.layers) == 4
    base = json.loads((out / "results/base.json").read_text())
    art = json.loads((out / "results/artifact.json").read_text())
    assert art["metrics"]["params_b"]["value"] < base["metrics"]["params_b"]["value"]
    assert manifest["diff_summary"]["removed_tensors"] > 0


def test_skip_eval_writes_patch_only(tiny_model_dir, tmp_path):
    recipe = tiny_recipe(tiny_model_dir, "narrow")
    out = run(recipe, tmp_path / "run", skip_eval=True, log=_quiet)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["gate_passed"] is False and manifest["eval_skipped"] is True
    assert (out / "artifact" / "patch.safetensors").exists()
    assert not (out / "README.md").exists()


def test_diff_identical_is_empty(tiny_model_dir):
    report = diff(tiny_model_dir, tiny_model_dir)
    assert report.summary()["changed_tensors"] == 0
    assert report.summary()["unchanged_tensors"] > 0
