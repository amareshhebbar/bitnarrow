import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import kit
import yaml

TASKS = {
    "mmlu": {"limit": 40, "metric": "acc", "fewshot": None},
    "gsm8k": {"limit": 400, "metric": "exact_match", "fewshot": None},
    "arc_challenge": {"limit": 500, "metric": "acc_norm", "fewshot": None},
    "hellaswag": {"limit": 500, "metric": "acc_norm", "fewshot": None},
}
FILTERS = ("strict-match", "none")


def sample_score(sample: dict, metric: str) -> float | None:
    if sample.get("filter") not in (None, *FILTERS):
        return None
    value = sample.get(metric)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--variant", choices=["base", "narrow"], required=True)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--batch-size", type=int, default=int(os.environ.get("EVAL_BATCH", 16)))
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--scale", type=float, default=float(os.environ.get("CAP_SCALE", 1.0)))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from bitnarrow.models import load_model, load_tokenizer
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    kit.seed_everything(args.seed)
    m = yaml.safe_load((HERE / "models.yaml").read_text(encoding="utf-8"))[args.tag]
    dtype = args.dtype or m["dtype"]
    run_dir = Path(args.run_dir or f"runs/bench_{args.tag}")
    patch = str(run_dir / "artifact") if args.variant == "narrow" else None
    out_dir = HERE / "results" / args.tag / "raw"

    tok = load_tokenizer(m["model"])
    model = load_model(m["model"], dtype=dtype, load_in_4bit=True, patch=patch)
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=args.batch_size)
    rows, summary = [], {}
    for task in args.tasks.split(","):
        spec = TASKS[task]
        limit = max(1, int(spec["limit"] * args.scale))
        started = time.time()
        res = simple_evaluate(
            model=lm,
            tasks=[task],
            num_fewshot=spec["fewshot"],
            limit=limit,
            log_samples=True,
            random_seed=args.seed,
            numpy_random_seed=args.seed,
            torch_random_seed=args.seed,
            fewshot_random_seed=args.seed,
        )
        n_task = 0
        for sub, samples in (res.get("samples") or {}).items():
            for s in samples:
                score = sample_score(s, spec["metric"])
                if score is None:
                    continue
                rows.append(
                    {
                        "task": task,
                        "subtask": sub,
                        "doc_id": s["doc_id"],
                        "variant": args.variant,
                        "metric": spec["metric"],
                        "score": score,
                    }
                )
                n_task += 1
        summary[task] = {
            "limit_per_subtask": limit,
            "items": n_task,
            "seconds": round(time.time() - started, 1),
            "lm_eval_results": {k: v for k, v in (res.get("results") or {}).items() if k == task},
            "n_shot": res.get("n-shot"),
        }
        print(f"[capability] {args.tag} {args.variant} {task}: {n_task} items", flush=True)
    kit.write_jsonl(out_dir / f"capability_{args.variant}.jsonl", rows)
    kit.write_json(
        out_dir / f"capability_{args.variant}.json",
        {
            "model": m["model"],
            "variant": args.variant,
            "load": "bitsandbytes nf4",
            "tasks": summary,
            "env": kit.env_capture(["torch", "transformers", "bitsandbytes", "lm_eval", "bitnarrow"]),
        },
    )


if __name__ == "__main__":
    main()
