import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import kit
import report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default=report.REPO_ID)
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()
    kit.load_env(HERE.parent / ".env")
    kit.load_env(".env")
    results = HERE / "results"
    for name in ("verify.py", "report.py", "kit.py", "models.yaml", "prepare_data.py", "requirements.txt"):
        dst = results / "code" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(HERE / name, dst)
    report.build(results)
    sha = kit.hf_upload(results, args.repo_id, private=args.private)
    print(f"uploaded {args.repo_id} at {sha}")
    print(kit.headline_url(args.repo_id))


if __name__ == "__main__":
    main()
