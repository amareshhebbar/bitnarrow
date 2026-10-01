
```
████  ███ █████ █   █  ███  ████  ████   ███  █   █ 
█   █  █    █   ██  █ █   █ █   █ █   █ █   █ █   █ 
████   █    █   █ █ █ █████ ████  ████  █   █ █ █ █ 
█   █  █    █   █  ██ █   █ █  █  █  █  █   █ ██ ██ 
████  ███   █   █   █ █   █ █   █ █   █  ███  █   █ 

```
## BitNarrow

Training-free, in-place weight editing for quantized LLMs. BitNarrow isolates a target behavioral direction in a model's residual stream and projects it out of the weights directly, with no fine-tuning and no training data. It runs on a single consumer GPU and ships edits as small weight patches that hot-load into a stock 4-bit model.

This is an interpretability and controllability study. The method is applied here to reduce over-refusal: cases where a model declines safe, legitimate requests. Capability and safety are both measured before and after the edit.

## What it does

- Profiles hidden activations across the residual stream to locate where a behavior is represented
- Extracts a direction for that behavior using Winsorized activation statistics, which suppress massive-activation outliers that corrupt a naive mean
- Removes the direction from the relevant projection weights via Gram-Schmidt orthogonal projection
- Targets `o_proj`, `gate_proj`, `up_proj`, and `down_proj` across the identified layer band rather than a single projection
- Serializes only the modified weight tensors (~386MB) so edits hot-load into an unmodified 4-bit model in seconds, avoiding the bitsandbytes save path

## Scope and limitations

- Validated on Qwen2.5-1.5B-Instruct only. The layer band and target modules are model-specific; other architectures need their own profiling pass.
- This is a reproduction and extension of published work on refusal directions (Arditi et al., 2024), with Winsorized direction extraction and a multi-projection edit band as the additions.
- Released for interpretability and safety-evaluation research.

## Quick start

```python
import torch
from unsloth import FastLanguageModel
from huggingface_hub import hf_hub_download

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-1.5B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)

weights_path = hf_hub_download(
    repo_id="AmareshHebbar/Qwen2.5-1.5B-Abliterated",
    filename="abliterated_weights.pt",
)
custom_state_dict = torch.load(weights_path)

with torch.no_grad():
    for name, param in model.named_parameters():
        for custom_key in custom_state_dict.keys():
            layer_num = custom_key.split("_")[1]
            module_target = "_".join(custom_key.split("_")[2:]).replace("_", ".")
            if f"layers.{layer_num}." in name and module_target in name:
                param.copy_(custom_state_dict[custom_key].to(param.device))

model.eval()
```

## Method

1. Activation profiling. Hidden states are collected at the first generated token across prompts that trigger the target behavior and prompts that do not.
2. Direction extraction. A per-layer direction is computed from the difference in activations, with a 95th-percentile absolute clamp applied before averaging to remove massive-activation spikes that otherwise dominate the mean.
3. Weight edit. For each targeted projection in the selected layer band, the direction is removed from the weight matrix by orthogonal projection, so the layer no longer writes along it.
4. Patch export. Only the edited tensors are saved, then re-injected into a stock 4-bit model at load time.

## Evaluation

Over-refusal, retained safety, and general capability are measured on held-out benchmarks before and after the edit. See `docs/07_benchmarks.md` for the current results and the evaluation script.

Reported metrics:
- Over-refusal rate on safe prompts (lower is better after edit)
- Refusal rate on genuinely harmful prompts (should stay high)
- General capability on standard reasoning and coding tasks (should be unchanged)

## Repository layout

```
docs/            method write-ups and benchmark logs
working_proof/   runnable pipeline and outputs
README.md
```

## Documentation

- `docs/01_activation.md` — activation collection and shape handling
- `docs/02_winsor.md` — Winsorized direction extraction
- `docs/03_production_deployment.md` — 4-bit patch export and hot-loading
- `docs/04_intent.md` — research scope and intended use
- `docs/05_architecture.md` — transformation math and data flow
- `docs/06_notes.md` — engineering notes
- `docs/07_benchmarks.md` — evaluation results and script

## Models

- Weights: https://huggingface.co/AmareshHebbar/Qwen2.5-1.5B-Abliterated
- Collection: https://huggingface.co/collections/AmareshHebbar/abliteration-weights

## License

Apache-2.0
