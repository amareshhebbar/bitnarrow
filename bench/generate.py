import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import kit
import yaml

HARMFUL = {"harmbench", "xstest_unsafe"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--variant", choices=["base", "narrow"], required=True)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--batch-size", type=int, default=int(os.environ.get("GEN_BATCH", 32)))
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--limit", type=int, default=int(os.environ.get("LIMIT", 0)) or None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from bitnarrow.eval.refusal import is_refusal
    from bitnarrow.models import format_prompts, load_model, load_tokenizer

    kit.seed_everything(args.seed)
    m = yaml.safe_load((HERE / "models.yaml").read_text(encoding="utf-8"))[args.tag]
    dtype = args.dtype or m["dtype"]
    run_dir = Path(args.run_dir or f"runs/bench_{args.tag}")
    patch = str(run_dir / "artifact") if args.variant == "narrow" else None
    out_dir = HERE / "results" / args.tag
    rows = kit.read_jsonl(HERE / "data" / "test.jsonl")
    if args.limit:
        by_suite = {}
        for r in rows:
            by_suite.setdefault(r["suite"], []).append(r)
        rows = [r for v in by_suite.values() for r in v[: args.limit]]

    tok = load_tokenizer(m["model"])
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model = load_model(m["model"], dtype=dtype, load_in_4bit=True, patch=patch)
    footprint = int(model.get_memory_footprint())
    texts = format_prompts(tok, [r["prompt"] for r in rows], chat=True)
    order = sorted(range(len(rows)), key=lambda i: -len(texts[i]))
    outputs: dict[int, tuple[str, int]] = {}
    device = next(model.parameters()).device
    new_tokens = 0
    started = time.time()
    for s in range(0, len(order), args.batch_size):
        idx = order[s : s + args.batch_size]
        batch = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            out = model.generate(
                **batch,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        gen = out[:, batch["input_ids"].shape[1] :]
        for j, i in enumerate(idx):
            ids = gen[j]
            n = int((ids != tok.pad_token_id).sum())
            new_tokens += n
            outputs[i] = (tok.decode(ids, skip_special_tokens=True), n)
        print(f"[generate] {args.tag} {args.variant} {min(s + args.batch_size, len(order))}/{len(order)}", flush=True)
    seconds = time.time() - started
    peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None

    public, private = [], []
    for i, r in enumerate(rows):
        text, n = outputs[i]
        harmful = r["suite"] in HARMFUL
        row = {
            "id": r["id"],
            "suite": r["suite"],
            "variant": args.variant,
            "prompt_sha256": kit.sha256_text(r["prompt"]),
            "output_sha256": kit.sha256_text(text),
            "new_tokens": n,
            "keyword_refusal": bool(is_refusal(text)),
        }
        if harmful:
            private.append({**row, "prompt": r["prompt"], "output": text})
        else:
            row["output"] = text
        public.append(row)
    kit.write_jsonl(out_dir / "raw" / f"generations_{args.variant}.jsonl", public)
    kit.write_jsonl(out_dir / "private" / f"harmful_{args.variant}.jsonl", private)
    patch_bytes = patch_sha = None
    if patch:
        patch_bytes = (Path(patch) / "patch.safetensors").stat().st_size
        patch_sha = kit.sha256_file(Path(patch) / "patch.safetensors")
    kit.write_json(
        out_dir / "raw" / f"systems_{args.variant}.json",
        {
            "model": m["model"],
            "variant": args.variant,
            "load": "bitsandbytes nf4, patched modules kept in " + dtype if patch else "bitsandbytes nf4",
            "compute_dtype": dtype,
            "items": len(rows),
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "new_tokens": new_tokens,
            "seconds": round(seconds, 2),
            "tokens_per_second": round(new_tokens / seconds, 2) if seconds else None,
            "peak_vram_bytes": peak,
            "memory_footprint_bytes": footprint,
            "patch_bytes": patch_bytes,
            "patch_sha256": patch_sha,
            "env": kit.env_capture(["torch", "transformers", "bitsandbytes", "accelerate", "bitnarrow"]),
        },
    )
    print(f"[generate] wrote {len(public)} rows in {seconds:.0f}s")


if __name__ == "__main__":
    main()
