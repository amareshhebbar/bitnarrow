from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from bitnarrow.models import apply_patch, load_model, load_tokenizer, read_patch, write_patch
from bitnarrow.surgery import narrow, prune
from conftest import tiny_recipe

pytestmark = pytest.mark.torch


def test_projection_math():
    torch.manual_seed(0)
    r = torch.nn.functional.normalize(torch.randn(16), dim=0)
    w_out = torch.randn(16, 40)
    w_in = torch.randn(40, 16)
    for s in (0.5, 1.0):
        writer = narrow.project_weight(w_out, r, "writer", s)
        reader = narrow.project_weight(w_in, r, "reader", s)
        assert torch.allclose(r @ writer, (1 - s) * (r @ w_out), atol=1e-5)
        assert torch.allclose(reader @ r, (1 - s) * (w_in @ r), atol=1e-5)
    full = narrow.project_weight(w_out, r, "writer", 1.0)
    x = torch.randn(40)
    assert abs(float(r @ (full @ x))) < 1e-4


def test_winsorized_mean_resists_outliers():
    x = torch.zeros(100, 4)
    x[0, 0] = 1e6
    assert float(narrow.winsorized_mean(x, 0.95)[0]) < 1.0
    assert float(narrow.winsorized_mean(x, 1.0)[0]) > 1e3


def test_direction_recovers_planted_signal():
    torch.manual_seed(0)
    d = 24
    planted = torch.nn.functional.normalize(torch.randn(d), dim=0)
    neg = torch.randn(5, 64, d)
    pos = torch.randn(5, 64, d)
    pos[3] += 4.0 * planted
    dirs, scores = narrow.layer_directions(pos, neg, 0.95)
    assert int(scores.argmax()) == 3
    assert float(dirs[3] @ planted) > 0.9
    band, cos = narrow.select_band(dirs, 3, 4, "auto", 0.99)
    assert band == [2]
    assert narrow.select_band(dirs, 3, 4, [0, 1], 0.5)[0] == [0, 1]
    with pytest.raises(ValueError):
        narrow.select_band(dirs, 3, 4, [2, 9], 0.5)


def test_narrow_end_to_end(tiny_model_dir, tmp_path):
    recipe = tiny_recipe(tiny_model_dir, "narrow")
    cfg = narrow.NarrowConfig.from_params({**recipe.params, "band": "all"})
    tok = load_tokenizer(str(tiny_model_dir))
    model = load_model(str(tiny_model_dir), dtype="float32")
    direction = narrow.find_direction(model, tok, cfg, chat=False)
    assert 1 <= direction.layer <= 6
    assert abs(float(direction.vector.norm()) - 1.0) < 1e-4
    patch = narrow.build_patch(model, direction, cfg, 1.0)
    assert any(k.endswith("down_proj.weight") for k in patch)
    assert any(k.endswith("gate_proj.weight") for k in patch)
    r = direction.vector
    for name, w in patch.items():
        if name.endswith(("o_proj.weight", "down_proj.weight")):
            assert float((r @ w.float()).abs().max()) < 1e-4
        if name.endswith(("gate_proj.weight", "up_proj.weight")):
            assert float((w.float() @ r).abs().max()) < 1e-4
    meta = narrow.patch_meta(direction, cfg, 1.0, str(tiny_model_dir), None)
    out = write_patch(patch, meta, tmp_path / "patch")
    tensors, loaded_meta = read_patch(out)
    assert loaded_meta["base_model"] == str(tiny_model_dir)
    fresh = load_model(str(tiny_model_dir), dtype="float32", patch=out)
    for name, value in tensors.items():
        assert torch.equal(fresh.get_parameter(name).cpu(), value)
    applied = apply_patch(model, tensors)
    assert sorted(applied) == sorted(tensors)


def test_patch_shape_mismatch(tiny_model_dir):
    model = load_model(str(tiny_model_dir), dtype="float32")
    name = "model.layers.0.mlp.down_proj.weight"
    with pytest.raises(ValueError):
        apply_patch(model, {name: torch.zeros(3, 3)})


def test_prune_removes_and_generates(tiny_model_dir, tmp_path):
    recipe = tiny_recipe(tiny_model_dir, "prune")
    cfg = prune.PruneConfig.from_params(recipe.params)
    tok = load_tokenizer(str(tiny_model_dir))
    model = load_model(str(tiny_model_dir), dtype="float32")
    decision = prune.choose(model, tok, cfg, chat=False)
    assert len(decision["remove"]) == 2
    assert decision["remove"][-1] < 5
    assert prune.remove_layers(model, decision["remove"]) == 4
    assert model.config.num_hidden_layers == 4
    assert [layer.self_attn.layer_idx for layer in model.model.layers] == [0, 1, 2, 3]
    batch = tok(["write a haiku about rain ."], return_tensors="pt").to(model.device)
    out = model.generate(**batch, max_new_tokens=5, min_new_tokens=5, do_sample=False, pad_token_id=0)
    assert out.shape[1] == batch["input_ids"].shape[1] + 5
    model.save_pretrained(tmp_path / "pruned")
    reloaded = load_model(str(tmp_path / "pruned"), dtype="float32")
    assert len(reloaded.model.layers) == 4


def test_prune_layerwise_and_limits():
    hidden = torch.randn(7, 10, 8)
    scores = prune.score_layers(hidden, protect_last=True)
    assert [i for i, _ in scores] == list(range(5))
    with pytest.raises(ValueError):
        prune.score_blocks(hidden, 6, protect_last=True)
    with pytest.raises(ValueError):
        prune.PruneConfig.from_params({"calibration": ["x"], "n_remove": 1, "mode": "random"})


def test_patch_meta_is_json(tiny_model_dir, tmp_path):
    recipe = tiny_recipe(tiny_model_dir, "narrow")
    cfg = narrow.NarrowConfig.from_params(recipe.params)
    d = narrow.Direction(vector=torch.ones(32) / 32**0.5, layer=3, band=[1, 2], scores=[0.0] * 7, band_cosines=[1.0] * 6)
    json.dumps(narrow.patch_meta(d, cfg, 0.5, "m", None))
