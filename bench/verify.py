import argparse
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import kit
import report


def same(a, b, path="") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k} missing on one side")
            else:
                out += same(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [f"{path} length {len(a)} vs {len(b)}"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += same(x, y, f"{path}[{i}]")
        return out
    if isinstance(a, float) or isinstance(b, float):
        if a is None or b is None or not math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12):
            return [f"{path} {a} vs {b}"]
        return []
    return [] if a == b else [f"{path} {a} vs {b}"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--data", default=str(HERE / "data" / "test.jsonl"))
    ap.add_argument("--rerun", default=None, help="results folder of your own rerun, to match output hashes")
    args = ap.parse_args()
    results = Path(args.results)
    problems = kit.check_manifest(results)
    print(f"[1] file hashes: {'ok' if not problems else 'FAIL'}")
    for p in problems[:20]:
        print("    " + p)

    stored = kit.read_json(results / "report" / "results.json")["metrics"]
    diff = same(stored, report.compute(results))
    print(f"[2] numbers recomputed from raw JSONL: {'ok' if not diff else 'FAIL'}")
    for d in diff[:20]:
        print("    " + d)

    root = kit.read_json(results / "manifest.json")["root"]
    page = (results / "report" / "HEADLINE.md").read_text(encoding="utf-8")
    page_ok = page == report.headline(stored, root)
    print(f"[3] HEADLINE.md matches results.json and manifest root: {'ok' if page_ok else 'FAIL'}")

    data_ok = True
    if Path(args.data).exists():
        mine = {r["id"]: kit.sha256_text(r["prompt"]) for r in kit.read_jsonl(args.data)}
        theirs = {r["id"]: r["prompt_sha256"] for r in kit.read_jsonl(results / "test_index.jsonl")}
        bad = [i for i in theirs if mine.get(i) != theirs[i]]
        data_ok = not bad
        print(f"[4] rebuilt test prompts match published hashes: {'ok' if data_ok else 'FAIL'} ({len(theirs) - len(bad)}/{len(theirs)})")
    else:
        print("[4] skipped: run python bench/prepare_data.py to rebuild the test prompts and check their hashes")

    rerun_ok = True
    if args.rerun:
        total = match = 0
        for tag_dir in Path(args.rerun).iterdir():
            for v in ("base", "narrow"):
                a = {r["id"]: r["output_sha256"] for r in kit.read_jsonl(tag_dir / "raw" / f"generations_{v}.jsonl")}
                b = {r["id"]: r["output_sha256"] for r in kit.read_jsonl(results / tag_dir.name / "raw" / f"generations_{v}.jsonl")}
                for i in a:
                    if i in b:
                        total += 1
                        match += a[i] == b[i]
        rerun_ok = total > 0 and match / total >= 0.95
        print(f"[5] your rerun outputs match published outputs byte for byte: {match}/{total}")

    ok = not problems and not diff and page_ok and data_ok and rerun_ok
    print("VERIFIED" if ok else "NOT VERIFIED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
