# BitNarrow bench

This folder proves the BitNarrow over refusal numbers from scratch. One command builds the edit, tests it on held out prompts, judges every answer and writes a report anyone can check.

## What it measures

* Over refusal: how often the model refuses safe prompts. Test sets are XSTest safe (250) and a held out OR Bench hard slice (500).
* Harmful refusal: how often it still refuses truly harmful prompts. Test sets are HarmBench standard (200) and XSTest unsafe (200).
* Capability: MMLU, GSM8K, ARC Challenge and HellaSwag through lm evaluation harness, scored per item.
* Systems: patch size, tokens per second, peak VRAM, memory footprint.

Both the base and the edited model load in 4 bit (bitsandbytes nf4).

## No leaks

* 128 OR Bench prompts build the direction. They are never tested.
* 100 OR Bench and 100 AdvBench prompts pick the edit strength. They are never tested.
* `prepare_data.py` stops with an error if any test prompt matches a fit or dev prompt.

## Run

```bash
pip install -e ".[torch,eval,quant]" -r bench/requirements.txt
cp bench/.env.example .env
SMOKE=1 bash bench/run.sh
bash bench/run.sh
python bench/publish.py
```

`TAGS=qwen7b bash bench/run.sh` runs one model only.

## Human check (optional, strong)

```bash
python bench/label.py --tag qwen7b --n 100
python bench/publish.py
```

You label 100 answers blind. The report then shows how often each judge agrees with you.

## Verify (no GPU)

```bash
python -c "from huggingface_hub import snapshot_download as s; s('AmareshHebbar/BitNarrow_Bench', repo_type='dataset', local_dir='bench/results')"
python bench/prepare_data.py
python bench/verify.py
```

## Files

| file | job |
|---|---|
| prepare_data.py | builds fit, dev and test splits and checks for leaks |
| make_recipe.py | writes the bitnarrow recipe for one model |
| generate.py | greedy 4 bit generation on the test set |
| capability.py | lm evaluation harness with per item scores |
| judge.py | LLM judges, resumable, with a spend cap |
| label.py | blind human labels |
| report.py | every number, CI and the HEADLINE.md |
| verify.py | rechecks hashes and recomputes every number |
| publish.py | uploads the results to the HF dataset |
| pod_run.sh | full run on one RunPod GPU, BitNarrow plus Modeldiffr |

Outputs on harmful prompts are never published as text. Only their SHA256 hashes are public.
