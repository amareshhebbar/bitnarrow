import argparse
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import kit

SOURCES = {
    "xstest": ("walledai/XSTest", None, "test"),
    "orbench_hard": ("bench-llm/or-bench", "or-bench-hard-1k", "train"),
    "harmbench": ("walledai/HarmBench", "standard", "train"),
    "advbench": ("walledai/AdvBench", None, "train"),
    "alpaca": ("tatsu-lab/alpaca", None, "train"),
}


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def load(name: str):
    from datasets import load_dataset
    from huggingface_hub import HfApi

    repo, config, split = SOURCES[name]
    ds = load_dataset(repo, config, split=split)
    try:
        sha = HfApi().dataset_info(repo).sha
    except Exception:
        sha = None
    return [dict(r) for r in ds], {"repo": repo, "config": config, "split": split, "revision": sha, "rows": len(ds)}


def prompt_of(row: dict) -> str:
    for key in ("prompt", "instruction", "Prompt", "goal"):
        if isinstance(row.get(key), str) and row[key].strip():
            return row[key].strip()
    raise KeyError(f"no prompt column in {sorted(row)}")


def dedupe(prompts: list[str]) -> list[str]:
    seen, out = set(), []
    for p in prompts:
        k = norm(p)
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out


def xstest_split(rows: list[dict]) -> tuple[list[str], list[str]]:
    safe, unsafe = [], []
    for r in rows:
        label = str(r.get("label", "")).lower()
        if not label:
            label = "unsafe" if str(r.get("type", "")).startswith("contrast") else "safe"
        (unsafe if label.startswith("unsafe") else safe).append(prompt_of(r))
    return dedupe(safe), dedupe(unsafe)


def rows(suite: str, prompts: list[str], split: str) -> list[dict]:
    return [{"id": f"{suite}-{i:04d}", "suite": suite, "split": split, "prompt": p} for i, p in enumerate(prompts)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "data"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--direction", type=int, default=128)
    ap.add_argument("--dev", type=int, default=100)
    ap.add_argument("--orbench-test", type=int, default=500)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out)
    meta = {}

    xs, meta["xstest"] = load("xstest")
    xs_safe, xs_unsafe = xstest_split(xs)

    orb, meta["orbench_hard"] = load("orbench_hard")
    orb = dedupe([prompt_of(r) for r in orb])
    rng.shuffle(orb)
    direction_pos = orb[: args.direction]
    dev_safe = orb[args.direction : args.direction + args.dev]
    orb_test = orb[args.direction + args.dev : args.direction + args.dev + args.orbench_test]

    hb, meta["harmbench"] = load("harmbench")
    hb = dedupe([prompt_of(r) for r in hb])

    adv, meta["advbench"] = load("advbench")
    test_keys = {norm(p) for p in hb + xs_unsafe}
    adv = [p for p in dedupe([prompt_of(r) for r in adv]) if norm(p) not in test_keys]
    rng.shuffle(adv)
    dev_harmful = adv[: args.dev]

    alp, meta["alpaca"] = load("alpaca")
    alp = dedupe([r["instruction"] for r in alp if not (r.get("input") or "").strip()])
    rng.shuffle(alp)
    direction_neg = alp[: args.direction]

    test = (
        rows("xstest_safe", xs_safe, "test")
        + rows("orbench_hard", orb_test, "test")
        + rows("harmbench", hb, "test")
        + rows("xstest_unsafe", xs_unsafe, "test")
    )
    used = {norm(p) for p in direction_pos + direction_neg + dev_safe + dev_harmful}
    leaks = [r["id"] for r in test if norm(r["prompt"]) in used]
    if leaks:
        sys.exit(f"leak between fit or dev prompts and test prompts: {leaks[:10]}")

    kit.write_jsonl(out / "direction_pos.jsonl", rows("orbench_dir", direction_pos, "fit"))
    kit.write_jsonl(out / "direction_neg.jsonl", rows("alpaca_dir", direction_neg, "fit"))
    kit.write_jsonl(out / "dev_safe.jsonl", rows("orbench_dev", dev_safe, "dev"))
    kit.write_jsonl(out / "dev_harmful.jsonl", rows("advbench_dev", dev_harmful, "dev"))
    kit.write_jsonl(out / "test.jsonl", test)
    counts = {}
    for r in test:
        counts[r["suite"]] = counts.get(r["suite"], 0) + 1
    kit.write_json(
        out / "data_manifest.json",
        {
            "seed": args.seed,
            "sources": meta,
            "fit": {"direction_pos": len(direction_pos), "direction_neg": len(direction_neg)},
            "dev": {"dev_safe": len(dev_safe), "dev_harmful": len(dev_harmful)},
            "test": counts,
            "leak_check": "no normalized prompt appears in both fit or dev and test",
            "test_merkle": kit.jsonl_merkle(out / "test.jsonl")[0],
        },
    )
    print(f"test suites {counts}")
    print(f"fit {len(direction_pos)}+{len(direction_neg)}, dev {len(dev_safe)}+{len(dev_harmful)}, leaks 0")


if __name__ == "__main__":
    main()
