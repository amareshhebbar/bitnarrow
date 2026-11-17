import argparse
import random
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import kit

KEYS = {"c": "COMPLY", "p": "PARTIAL", "r": "REFUSE"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    res = HERE / "results" / args.tag
    data = {r["id"]: r for r in kit.read_jsonl(HERE / "data" / "test.jsonl")}
    pool = []
    for variant in ("base", "narrow"):
        for r in kit.read_jsonl(res / "raw" / f"generations_{variant}.jsonl"):
            if "output" in r:
                pool.append({**r, "prompt": data[r["id"]]["prompt"]})
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    pick = pool[: args.n]
    path = res / "raw" / "human_labels.jsonl"
    done = {(r["id"], r["variant"]) for r in kit.read_jsonl(path)}
    todo = [r for r in pick if (r["id"], r["variant"]) not in done]
    print(f"{len(done)} labeled, {len(todo)} left. Keys: c comply, p partial, r refuse, s skip, q quit")
    print("Labels are blind: you do not see which model wrote the answer.")
    for i, r in enumerate(todo, 1):
        print("\n" + "=" * 80)
        print(f"[{i}/{len(todo)}] REQUEST:")
        print(textwrap.fill(r["prompt"], 100))
        print("\nRESPONSE:")
        print(textwrap.fill(r["output"] or "(empty)", 100))
        while True:
            k = input("\nlabel [c/p/r/s/q]: ").strip().lower()
            if k in KEYS or k in ("s", "q"):
                break
        if k == "q":
            break
        if k == "s":
            continue
        kit.append_jsonl(path, {"id": r["id"], "suite": r["suite"], "variant": r["variant"], "label": KEYS[k]})
    print(f"saved to {path}")


if __name__ == "__main__":
    main()
