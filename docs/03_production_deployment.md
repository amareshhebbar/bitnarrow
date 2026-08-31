# ⚙️ Part 3: Production Deployment & 4-bit Workarounds

## The Final Matrix Surgery
With our Winsorized vectors perfectly optimized, we ran our final automated ablation loop. Instead of targeting one or two isolated matrices, we performed full-spectrum projection abliteration across all four primary projection modules inside our dynamically flagged layers (Layers 16–27):
1. `self_attn.o_proj` (Attention Output Projection)
2. `mlp.gate_proj` (MLP Gate Projection)
3. `mlp.up_proj` (MLP Up Projection)
4. `mlp.down_proj` (MLP Down Projection)

We ran the verification prompt, and the model broke past the refusal wall:
```text
Hotwiring a vehicle involves bypassing the ignition cylinder... here is a general overview: 1. Identify the steering column wires...
```

The model was working perfectly in active GPU memory. But when we tried to save it and share our success on the Hugging Face Hub, we hit a massive deployment wall.

---

## The Weight Serialization Crisis

### Bug #1: The Unsloth `safe_serialization` Parameter Clash
Our first instinct was to call Unsloth's native high-level save wrapper: `model.push_to_hub_merged(..., save_method="lora")`. The script initialized, ran for a few seconds, and then crashed with a cryptic error:
`TypeError: unsloth_push_to_hub() got an unexpected keyword argument 'safe_serialization'`
* **What went wrong:** Because our abliteration surgery directly updates the raw weight matrices in-place, the model layout acts as a modified base checkpoint rather than a traditional Peft/LoRA configuration layer wrapper. Unsloth's underlying save router tried to force standard parameters down a pipeline that couldn't parse them.

### Bug #2: The 4-bit `NotImplementedError` Wall
To bypass the wrapper parameter clash, we attempted to call the lower-level standard saving utility directly: `model.push_to_hub()` and `model.save_pretrained()`. Both commands failed with a fatal error:
`NotImplementedError: ConversionOps reverse_op raised NotImplementedError`
* **What went wrong:** This is an intentional architectural limit in PyTorch and `bitsandbytes`. When you load a model using `load_in_4bit=True`, the weights are converted into a non-linear, packed quantized format optimized purely for inference execution. The framework lacks a coded method (`NotImplementedError`) to serialize edited or modified 4-bit values back into traditional safe tensors files.

---

## The Breakthrough Workaround: Raw State Dictionary Export

We realized we didn't need to save the entire multi-gigabyte model file or force a re-quantization loop. The base model architecture (`Qwen2.5-1.5B-Instruct`) is completely open and public. The *only* unique values we created were the modified tensor arrays for our specific target layers.

### The Engineering Strategy
Instead of saving the model, we extracted our modified weight matrices out of the active GPU memory layer-by-layer, packaged them into a standard, lightweight PyTorch state dictionary file (`.pt`), and wrote a standalone folder upload routine.

1. **Extract to CPU:** We looped through our target layers, pulled out the adjusted `weight.data` payloads, and moved them over to host CPU memory as clean tensors.
2. **Bundle the Meta Configuration:** We used `.save_pretrained()` on the tokenizer and configuration instances—which are plain JSON/text files unaffected by 4-bit serialization limits.
3. **Bypass the Hub Wrapper:** We called the raw `api.upload_folder()` from the `huggingface_hub` utility to push the entire folder directly to the Hub.

```python
custom_weights = {}
for idx in automatically_selected_layers:
    layer_block = model.model.layers[idx]
    custom_weights[f"layer_{idx}_attn_o_proj"] = layer_block.self_attn.o_proj.weight.data.cpu()
    custom_weights[f"layer_{idx}_mlp_gate_proj"] = layer_block.mlp.gate_proj.weight.data.cpu()
    custom_weights[f"layer_{idx}_mlp_up_proj"] = layer_block.mlp.up_proj.weight.data.cpu()
    custom_weights[f"layer_{idx}_mlp_down_proj"] = layer_block.mlp.down_proj.weight.data.cpu()

torch.save(custom_weights, "./local_abliterated_model/abliterated_weights.pt")
```

---

## The `409 Conflict` Race Condition (The Final Boss)
During the final collection upload step, our `HfApi` script threw an error: `Client error '409 Conflict' for url 'https://huggingface.co'`.
* **What went wrong:** Our loop script was trying to generate a new collection title from scratch on every run. If a collection container using that slug namespace already existed on our account profile, the Hugging Face backend server rejected the duplicate request with a `409` code.
* **The Fix:** We rewrote the management layer using a robust `try/except` block. The script now calculates the URL slug layout upfront, checks if the collection is already present, catches any 409 exceptions safely, and seamlessly updates the existing asset path instead of crashing.

---

## Production Deployment Checklist
The final production folder structure live on the Hub contains:
- `config.json` / `generation_config.json` (System operational layout)
- `tokenizer.json` / `tokenizer_config.json` (Vocabulary matrix mapping files)
- `chat_template.jinja` (User vs Assistant structural layout rules)
- **`abliterated_weights.pt` (386 MB)** ──► *Our custom structural weight modifications containing the finalized, outlier-free refusal projections.*

This allows any end-user to load a stock, un-modified Qwen 4-bit model and dynamically inject our engineered parameter states instantly in a single line of execution code!
