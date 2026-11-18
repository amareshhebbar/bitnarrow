#!/usr/bin/env bash
set -euo pipefail
W="${W:-/workspace}"
cd "$W"
export HF_HOME="$W/hf" PIP_ROOT_USER_ACTION=ignore TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -f "$W/.env" ] || { echo "put your filled .env at $W/.env first"; exit 1; }
set -a; . "$W/.env"; set +a

for r in bitnarrow modeldiffr; do
  [ -d "$r/.git" ] || git clone "https://github.com/amareshhebbar/$r.git"
  cp "$W/.env" "$r/.env"
  [ -f "$r/bench/run.sh" ] || { echo "$r/bench is missing: push the bench folder to GitHub first (see README step 3)"; exit 1; }
done

python -m pip install -q -U pip
python -m pip install -q -e "bitnarrow[torch,eval,quant]" -r bitnarrow/bench/requirements.txt
python -m pip install -q -e "modeldiffr[hf,tasks,batch]" -r modeldiffr/bench/requirements.txt
python -c "import torch, transformers, bitsandbytes, lm_eval; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

if [ "${SMOKE:-0}" = "1" ]; then
  (cd bitnarrow && SMOKE=1 bash bench/run.sh)
  (cd modeldiffr && SMOKE=1 bash bench/run.sh)
  rm -rf bitnarrow/bench/results bitnarrow/runs/bench_smoke modeldiffr/bench/results modeldiffr/bench/spec.yaml
  echo "smoke test passed and its outputs were removed. Now run: bash bitnarrow/bench/pod_run.sh"
  exit 0
fi

(cd bitnarrow && bash bench/run.sh) 2>&1 | tee "$W/bitnarrow.log"
python -c "import torch; torch.cuda.empty_cache()"
(cd modeldiffr && BITNARROW_EXPORTS="$W/bitnarrow/exports" bash bench/run.sh) 2>&1 | tee "$W/modeldiffr.log"

if [ "${PUBLISH:-1}" = "1" ]; then
  (cd bitnarrow && python bench/publish.py)
  (cd modeldiffr && python bench/publish.py)
  python - <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = "AmareshHebbar/BitNarrow_Bench_private"
api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
for tag in os.listdir("bitnarrow/bench/results"):
    p = f"bitnarrow/bench/results/{tag}/private"
    if os.path.isdir(p):
        api.upload_folder(folder_path=p, path_in_repo=f"{tag}/private", repo_id=repo, repo_type="dataset")
print("private harmful outputs saved to", repo)
PY
  for t in qwen7b llama8b; do
    [ -d "bitnarrow/runs/bench_$t/artifact" ] && (cd bitnarrow && bitnarrow publish "runs/bench_$t" || echo "patch publish for $t refused, see the gate in runs/bench_$t/gate.json")
  done
fi
echo "ALL DONE. You can stop the pod now."
