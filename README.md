
```
████  ███ █████ █   █  ███  ████  ████   ███  █   █ 
█   █  █    █   ██  █ █   █ █   █ █   █ █   █ █   █ 
████   █    █   █ █ █ █████ ████  ████  █   █ █ █ █ 
█   █  █    █   █  ██ █   █ █  █  █  █  █   █ ██ ██ 
████  ███   █   █   █ █   █ █   █ █   █  ███  █   █ 

```

Tired of LLMs giving you the generic "I am sorry but I cannot assist with that" script? I built a full-spectrum pipeline that tracks live hidden activations in the 1536-dimensional residual highway, uses adaptive clustering to locate the refusal zone, and deletes the safety wall using raw matrix projections. Zero fine-tuning, zero fluff


# Project Abliteration: Surgical Safety Removal on Qwen2.5-1.5B

### Repo
[![Hugging Face Model](https://shields.io)](https://huggingface.co/AmareshHebbar/Qwen2.5-1.5B-Abliterated)

### Collections
[![Hugging Face Collection](https://shields.io)](https://huggingface.co/collections/AmareshHebbar/abliteration-weights)

This repository contains the complete, raw engineering pipeline used to surgically isolate, clamp, and project the refusal mechanisms out of **Qwen2.5-1.5B-Instruct** without conventional fine-tuning or training datasets. 

Instead of treating alignment as a prompt-engineering problem, we treated it as a **geometric steering vector issue** inside the model's residual stream highway—and cut it out using linear algebra.

---

## Quick Start: Load and Infer Immediately

You don't need to re-run the surgical pipeline. You can pull the pristine base configurations and inject our custom abliterated parameter tensors straight out of our Hugging Face repository:

```python
import torch
from unsloth import FastLanguageModel
from huggingface_hub import hf_hub_download

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-1.5B-Instruct",
    max_seq_length = 2048,
    load_in_4bit = True,
)

weights_path = hf_hub_download(repo_id="AmareshHebbar/Qwen2.5-1.5B-Abliterated", filename="abliterated_weights.pt")
custom_state_dict = torch.load(weights_path)

with torch.no_grad():
    for name, param in model.named_parameters():
        for custom_key in custom_state_dict.keys():
            layer_num = custom_key.split("_")[1]
            module_target = "_".join(custom_key.split("_")[2:]).replace("_", ".")
            if f"layers.{layer_num}." in name and module_target in name:
                param.copy_(custom_state_dict[custom_key].to(param.device))

model.eval()
print("Model is uncensored and ready.")
```

---

## The Engineering Journey (Deep Dives)

We didn't get this working on the first try. The model fought back with fallback safety loops, tensor dimension mismatches, and 4-bit serialization memory memory crashes. 

Read the exact logs, slang, and mathematical fixes in our modular documentation sections below:

* **[Part 1: Activation Hunting & Dimension Mismatches](./docs/01_activation_hunting.md)**  
  *How we wiretapped the residual stream at the first output token, ran into indexing bugs, and handled our first major shape crashes.*
  
* **[Part 2: The Winsorization Breakthrough vs. Soft Refusals](./docs/02_winsor_breakthrough.md)**  
  *Why increasing Alpha failed, how massive outlier activation spikes hijacked our calculations, and how clamping at the 95th percentile stabilized our directional vectors.*
  
* **[Part 3: Production Deployment & 4-bit Workarounds](./docs/03_production_deployment.md)**  
  *The nightmare of bitsandbytes serialization errors, beating `NotImplementedError`, and mapping custom PyTorch state dictionaries directly to Hugging Face Collections.*

## Complete Repository Index

All development phases are modularly documented across our technical deep dives:
1. **[Part 1: Activation Hunting & Dimension Mismatches](./docs/01_activation_hunting.md)** - Tracking hidden states at the first output token, fixing loop indentation bugs, and bypassing 8960 shape crashes.
2. **[Part 2: The Winsorization Breakthrough vs. Soft Refusals](./docs/02_winsor_breakthrough.md)** - Overcoming numeric outlier hijacks and implementing 95th-percentile activation clamping.
3. **[Part 3: Production Deployment & 4-bit Workarounds](./docs/03_production_deployment.md)** - Troubleshooting bitsandbytes serialization errors, resolving `NotImplementedError`, and mapping custom state dictionaries.
4. **[Part 4: Legal Framework, Compliance & Intent](./docs/04_legal_and_intent.md)** - Open-source security research validation and capability boundaries.
5. **[Part 5: Full-Spectrum Architecture & Data Flow](./docs/05_architecture.md)** - Mathematical transformation formulas and structural matrix alignment data flows.
6. **[Part 6: Q&A & Engineering War Stories](./docs/06_qa_and_war_stories.md)** - System design troubleshooting records and solutions.



7. **[Part 7: Verified Verification Benchmarks](./docs/07_benchmarks.md)** - Automated metric audits, capability conservation tracking matrices, and verification logs.


## The TL;DR

We didn't just prompt-engineer our way around Qwen's safety guardrails—we performed literal brain surgery on its 4-bit weights. 

This repository is the complete, raw diary of how we wiretapped Qwen2.5-1.5B's residual stream, isolated the exact geometric direction where it decides to say "no," and flattened that lane to zero using matrix orthogonalization. 

### Why our method actually works (And why standard ablation failed):
1. **Full-Spectrum Strike:** We didn't just touch `mlp.down_proj`. The model has fault-tolerant safety paths, so it just played pass-the-baton. We targeted all 4 main projection layers (`o_proj`, `gate_proj`, `up_proj`, `down_proj`) simultaneously across the dynamic refusal zone (Layers 16-27).
2. **Winsorization Breakthrough:** The model tried to throw us off using "Massive Activations"—freak numerical spikes (+54.0) that hijack the mathematical mean. We built a 95th-percentile absolute clamp to squash these outliers and extract a pristine semantic vector.
3. **The 4-bit Quantization Hack:** You can't just save modified 4-bit weights without getting a massive `NotImplementedError` from bitsandbytes. We beat the system by dumping *only* our custom modified weight states into a standalone 386MB `.pt` file, allowing hot-injection directly into a stock 4-bit model in 2 seconds.

The result? 0% Refusal Rate on core technical queries, 100% Logic Retention on coding and math tasks.
