#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f bench/.env ] && set -a && . bench/.env && set +a
[ -f .env ] && set -a && . ./.env && set +a

TAGS="${TAGS:-qwen7b llama8b}"
SMOKE="${SMOKE:-0}"
EXTRA_GEN=""
if [ "$SMOKE" = "1" ]; then
  TAGS="smoke"
  export LIMIT="${LIMIT:-10}" CAP_SCALE="${CAP_SCALE:-0.05}"
fi

python bench/prepare_data.py

for t in $TAGS; do
  echo "==== $t: build the edit with bitnarrow (dev split picks the strength)"
  python bench/make_recipe.py --tag "$t"
  bitnarrow run "bench/recipes/$t.yaml" --out "runs/bench_$t"
  mkdir -p "bench/results/$t/run"
  cp "runs/bench_$t/recipe.yaml" "runs/bench_$t/manifest.json" "bench/results/$t/run/"
  [ -f "runs/bench_$t/gate.json" ] && cp "runs/bench_$t/gate.json" "bench/results/$t/run/"
  [ -d "runs/bench_$t/results" ] && cp -r "runs/bench_$t/results" "bench/results/$t/run/dev_results"
  cp "runs/bench_$t/artifact/patch.json" "bench/results/$t/run/patch.json"

  echo "==== $t: generate on the held out test set, 4 bit"
  python bench/generate.py --tag "$t" --variant base
  python bench/generate.py --tag "$t" --variant narrow

  echo "==== $t: judge in the background while capability runs"
  python bench/judge.py --tag "$t" > "bench/results/$t/judge.log" 2>&1 &
  JPID=$!

  echo "==== $t: capability, 4 bit"
  python bench/capability.py --tag "$t" --variant base
  python bench/capability.py --tag "$t" --variant narrow
  wait "$JPID" || echo "judge exited with an error, see bench/results/$t/judge.log, rerun: python bench/judge.py --tag $t"

  if [ "${MATERIALIZE:-1}" = "1" ] && [ "$t" != "smoke" ]; then
    MODEL=$(python -c "import yaml;print(yaml.safe_load(open('bench/models.yaml'))['$t']['model'])")
    bitnarrow export "$MODEL" --patch "runs/bench_$t/artifact" --materialize "exports/${t}_narrow"
  fi
done

python bench/report.py
python bench/verify.py
echo "done. publish with: python bench/publish.py"
