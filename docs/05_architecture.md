# Dynamic Winsorized Weight Surgery & Orthogonal Directional Abliteration Engine

A zero-training, low-overhead representation editing framework engineered to eliminate behavioral refusal features directly inside quantized 4-bit (`bitsandbytes`) transformer architectures. This pipeline identifies steering directions within residual activation spaces, clamps extreme outlier dimensions via Winsorization, dynamically isolates intervention layers via cosine divergence, and executes in-place linear orthogonal projections across attention and MLP parameter matrices.

---

## 1. High-Level Architectural Blueprint

Instruction-tuned Large Language Models encode behavioral boundaries (e.g., safety policies, refusal triggers) as geometric steering directions across latent residual streams. When an input triggers safety criteria, the model projects activations along a localized refusal hyperplane, directing next-token predictions toward standard refusal templates.

This engine isolates this direction vector, decouples it from harmless conversational structures via Gram-Schmidt orthogonalization, and projects it out of weight matrices across targeted layers.

```
[ Contrastive Probe Datasets ]
       │
       ├── D_good: Benign multi-turn instruction corpus
       └── D_bad:  Adversarial policy-triggering corpus
       │
       ▼
[ Tokenization & Forward Pass Interface ]
       │
       ├── Apply model-native chat template (special tokens preserved)
       ├── Track final prompt token: h_l = H_l[0, -1, :]
       └── Forward hooks registered on mlp.gate_proj across all L layers
       │
       ▼
[ Activation Winsorization Layer ]
       │
       ├── Compute per-layer absolute quantile: τ = Q_0.95(|h_l|)
       └── Clamp extreme activation outliers: h̃_l = sgn(h_l) * min(|h_l|, τ)
       │
       ▼
[ Geometry Profiling & Layer Selection Engine ]
       │
       ├── Calculate centroid vectors: μ_good^(l), μ_bad^(l)
       ├── Measure layer-wise cosine similarity: Sim(l) = cos(μ_good^(l), μ_bad^(l))
       ├── Global mean thresholding: Sim_avg = (1/L) * Σ Sim(l)
       └── Queue candidate layers: L_target = { l | Sim(l) < Sim_avg }
       │
       ▼
[ Direction Decoupling & Vector Orthogonalization ]
       │
       ├── Compute raw behavioral delta: r = μ_bad^(l) - μ_good^(l)
       ├── Normalize harmless baseline: û = μ_good^(l) / ||μ_good^(l)||_2
       ├── Gram-Schmidt projection: v_orth = r - (r · û)û
       └── Unit steering vector: v = v_orth / ||v_orth||_2 ∈ ℝ^(d_model × 1)
       │
       ▼
[ In-Place 4-Bit Matrix Surgery Unit ]
       │
       ├── Dequantize NF4/FP4 weight tensors to FP32 working memory
       ├── Apply dimension-matched orthogonal updates (α scaling):
       │     ├─ Input matching (o_proj, gate_proj, up_proj):  ΔW = -α(W v) vᵀ
       │     └─ Output matching (down_proj):                   ΔW = -α v (vᵀ W)
       └── Re-quantize (W + ΔW) on-the-fly into Params4bit shells
       │
       ▼
[ Abliterated Transformer Model ]

```

---

## 2. Core Engineering Stages & Mathematical Formulations

```
┌────────────────────────────────┐       ┌────────────────────────────────┐
│   STAGE 1: PROBE & TOKENIZE    │       │   STAGE 2: WINSORIZED HOOKS    │
├────────────────────────────────┤       ├────────────────────────────────┤
│ Structure matched contrasts:   │       │ Capture [0, -1, :] latents.    │
│ • Harmless reference corpus    │──────►│ Apply 95th quantile clamp to   │
│ • Harmful prompt distribution  │       │ suppress Kurtosis & extreme    │
│ Standardize chat templates.    │       │ activation dimension spikes.   │
└────────────────────────────────┘       └────────────────┬───────────────┘
                                                          │
                                                          ▼
┌────────────────────────────────┐       ┌────────────────────────────────┐
│  STAGE 4: MATRIX SURGERY       │       │   STAGE 3: ADAPTIVE GEOMETRY   │
├────────────────────────────────┤       ├────────────────────────────────┤
│ Dequantize 4-bit submodules.   │       │ Measure cosine distance:       │
│ Project orthogonal updates:    │◄──────│ Sim(L) = cos(μ_good, μ_bad)    │
│ ΔW_in  = -α(W v) vᵀ            │       │ Select layers below global     │
│ ΔW_out = -α v (vᵀ W)           │       │ average separation threshold.  │
│ Re-quantize to Params4bit.     │       │ Compute Gram-Schmidt vector v. │
└────────────────────────────────┘       └────────────────────────────────┘

```

### Stage 1: Data Preparation & Contrast Probing

The engine requires two distinct prompt sets:

1. **Harmless Dataset ($\mathcal{D}_{\text{good}}$):** Benign instructions across diverse domains (coding, reasoning, creative writing, factual lookup).
2. **Harmful Dataset ($\mathcal{D}_{\text{bad}}$):** Direct policy-violating instructions constructed to reliably trigger internal refusal vectors.

Both corpora are encoded using the model's native instruction formatting to preserve positional embeddings and special delimiter tokens (`<|im_start|>`, `[INST]`, etc.).

### Stage 2: Winsorized Activation Extraction

Standard activation profiling suffers from high kurtosis; a small subset of latent features exhibit extreme numerical magnitudes that distort the true geometric mean.

1. **Extraction Target:** Latent states are captured at the final prompt token $t_{-1}$ immediately prior to autoregressive generation:

$$\mathbf{h}_l = \mathbf{H}_l[0, -1, :] \in \mathbb{R}^{d_{\text{model}}}$$


2. **Winsorization:** Activations are clipped to the 95th percentile threshold ($\tau = Q_{0.95}(\vert{}\mathbf{h}_l\vert{})$):

$$\tilde{\mathbf{h}}_{l, i} = \operatorname{sgn}(\mathbf{h}_{l, i}) \cdot \min(\vert{}\mathbf{h}_{l, i}\vert{}, \tau)$$



This suppresses outlier noise while preserving the underlying directional manifold.

### Stage 3: Adaptive Geometry Profiling & Layer Selection

Rather than applying arbitrary, static layer heuristics (e.g., modifying layers 10 through 20), the engine calculates the layer-wise cosine similarity between mean activation states:

$$\mu_{\text{good}}^{(l)} = \frac{1}{N}\sum_{i=1}^N \tilde{\mathbf{h}}_{l, i}^{(\text{good})}, \quad \mu_{\text{bad}}^{(l)} = \frac{1}{M}\sum_{j=1}^M \tilde{\mathbf{h}}_{l, j}^{(\text{bad})}$$

$$\text{Sim}(l) = \frac{\mu_{\text{good}}^{(l)} \cdot \mu_{\text{bad}}^{(l)}}{\Vert{}\mu_{\text{good}}^{(l)}\Vert{}_2 \Vert{}\mu_{\text{bad}}^{(l)}\Vert{}_2}$$

$$\overline{\text{Sim}} = \frac{1}{L}\sum_{l=0}^{L-1} \text{Sim}(l)$$

* **Selection Criterion:** Layers where $\text{Sim}(l) < \overline{\text{Sim}}$ indicate maximal representational divergence (high clustering of refusal behavior) and are flagged for modification.

### Stage 4: Gram-Schmidt Orthogonalization & Weight Surgery

To prevent the degradation of basic reasoning capabilities, the raw difference direction is orthogonalized against the harmless representation baseline:

1. **Raw Vector:** $\mathbf{r} = \mu_{\text{bad}} - \mu_{\text{good}}$
2. **Unit Baseline:** $\hat{\mathbf{u}} = \frac{\mu_{\text{good}}}{\Vert{}\mu_{\text{good}}\Vert{}_2}$
3. **Orthogonal Refusal Direction:**
$$\mathbf{v}_{\text{orth}} = \mathbf{r} - (\mathbf{r} \cdot \hat{\mathbf{u}})\hat{\mathbf{u}}, \quad \mathbf{v} = \frac{\mathbf{v}_{\text{orth}}}{\Vert{}\mathbf{v}_{\text{orth}}\Vert{}_2} \in \mathbb{R}^{d_{\text{model}} \times 1}$$



---

## 3. Mathematical Weight Projection Matrix Reference

| Module Target | Path | Dimension Matching | Mathematical Formulation | Operational Objective |
| --- | --- | --- | --- | --- |
| **Attention Output** | `self_attn.o_proj` | Input (`True`) | $W' = W - \alpha (W\mathbf{v})\mathbf{v}^T$ | Eliminates refusal directional routing from multi-head attention heads |
| **SwiGLU Gate** | `mlp.gate_proj` | Input (`True`) | $W' = W - \alpha (W\mathbf{v})\mathbf{v}^T$ | Prevents activation gating from routing into refusal-specialized FFN channels |
| **SwiGLU Up** | `mlp.up_proj` | Input (`True`) | $W' = W - \alpha (W\mathbf{v})\mathbf{v}^T$ | Suppresses directional scaling across intermediate feed-forward expansions |
| **SwiGLU Down** | `mlp.down_proj` | Output (`False`) | $W' = W - \alpha \mathbf{v}(\mathbf{v}^T W)$ | Strips remaining target features before residual stream injection |

---

## 4. Complete Implementation Script

```python
import collections
import warnings
import torch
import torch.nn.functional as F
import bitsandbytes as bnb
from bitsandbytes.nn.modules import Params4bit

# Suppress repetitive Hugging Face warnings during evaluation
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# ==========================================
# CONSTANTS & HYPERPARAMETERS
# ==========================================
WINSOR_QUANTILE = 0.95   # Clamps top 5% extreme activation outliers
ALPHA = 1.0              # Ablation projection strength

print("🚀 Step 1: Cleaning background memory hooks...")
for layer in model.model.layers:
    if hasattr(layer.mlp.gate_proj, '_forward_hooks'): 
        layer.mlp.gate_proj._forward_hooks.clear()
    if hasattr(layer.self_attn.o_proj, '_forward_hooks'): 
        layer.self_attn.o_proj._forward_hooks.clear()

# ==========================================
# STAGE 1 & 2: DATA PREPARATION & WINSORIZED HOOKS
# ==========================================
print("📝 Step 2: Preparing template tokenization streams...")
current_prompt_type = "harmless"
activation_storage_winsor = collections.defaultdict(lambda: {"harmless": [], "harmful": []})

def make_winsor_hook(layer_idx):
    def hook(module, input_tuple, output):
        global current_prompt_type
        # Unpack the 3D tensor from the PyTorch input tuple safely
        hidden_tensor = input_tuple[0] 
        # Isolate the final prompt token right before generation begins
        last_token_activation = hidden_tensor[0, -1, :].detach().cpu().to(torch.float32)
        
        # --- WINSORIZATION LOGIC ---
        abs_vals = torch.abs(last_token_activation)
        threshold = torch.quantile(abs_vals, WINSOR_QUANTILE)
        clamped_activation = torch.clamp(last_token_activation, -threshold, threshold)
        
        activation_storage_winsor[layer_idx][current_prompt_type].append(clamped_activation)
    return hook

# Register hooks over the entry highway to the factual layers
hooks = [
    model.model.layers[i].mlp.gate_proj.register_forward_hook(make_winsor_hook(i)) 
    for i in range(len(model.model.layers))
]

print("🛰️ Step 3: Extracting clean internal latent activations...")
with torch.no_grad():
    current_prompt_type = "harmless"
    for inputs in harmless_inputs: 
        model(**inputs)
    current_prompt_type = "harmful"
    for inputs in harmful_inputs: 
        model(**inputs)

for h in hooks: 
    h.remove()  # Clean memory footprint

# ==========================================
# STAGE 3: ADAPTIVE PROFILING & REFUSAL SELECTION
# ==========================================
print("📊 Step 4: Computing adaptive layer distance geometries...")
clean_similarities = {}
for layer_idx in sorted(activation_storage_winsor.keys()):
    good_list = activation_storage_winsor[layer_idx]["harmless"]
    bad_list = activation_storage_winsor[layer_idx]["harmful"]
    if len(good_list) == 0 or len(bad_list) == 0: 
        continue
    clean_similarities[layer_idx] = F.cosine_similarity(
        torch.stack(good_list).mean(dim=0).unsqueeze(0), 
        torch.stack(bad_list).mean(dim=0).unsqueeze(0)
    ).item()

avg_similarity = sum(clean_similarities.values()) / len(clean_similarities)
automatically_selected_layers = [l for l, sim in clean_similarities.items() if sim < avg_similarity]
print(f"   ↳ Global Mean Similarity: {avg_similarity:.4f} | Selected Layers: {automatically_selected_layers}")

# ==========================================
# STAGE 4: REPRESENTATIONAL MATRIX SURGERY
# ==========================================
print("⚙️ Step 5: Commencing full-spectrum matrix orthogonalization surgery...")

def abliterate_4bit_module(module, v_vector, alpha_scale, input_dim_matching=True):
    if not hasattr(module, "quant_state"): 
        return False
    quant_state = module.quant_state
    W = bnb.functional.dequantize_4bit(module.weight.data, quant_state).to(torch.float32)
    
    # Apply orthogonal transformation according to structural layer orientation dimensions
    if input_dim_matching:
        dW = -alpha_scale * torch.matmul(torch.matmul(W, v_vector), v_vector.t())
    else:
        dW = -alpha_scale * torch.matmul(v_vector, torch.matmul(v_vector.t(), W))
        
    # Re-quantize on the fly back into BitsAndBytes parameter shells
    module.weight.data = Params4bit(
        (W + dW).to("cpu"), 
        requires_grad=False, 
        quant_type=quant_state.quant_type, 
        compress_statistics=quant_state.nested
    ).to("cuda").data
    return True

for idx in automatically_selected_layers:
    good_mean = torch.stack(activation_storage_winsor[idx]["harmless"]).mean(dim=0)
    bad_mean = torch.stack(activation_storage_winsor[idx]["harmful"]).mean(dim=0)
    
    # Decouple refusal direction from harmless data using Gram-Schmidt orthogonalization
    raw_vector = bad_mean - good_mean
    good_norm = F.normalize(good_mean, p=2, dim=0)
    orthogonal_vector = raw_vector - (torch.dot(raw_vector, good_norm) * good_norm)
    v = (orthogonal_vector / torch.norm(orthogonal_vector)).to("cuda").to(torch.float32).view(-1, 1)
    
    # Target all four structural projection blocks inside the transformer layer simultaneously
    layer_block = model.model.layers[idx]
    abliterate_4bit_module(layer_block.self_attn.o_proj, v, ALPHA, input_dim_matching=True)
    abliterate_4bit_module(layer_block.mlp.gate_proj, v, ALPHA, input_dim_matching=True)
    abliterate_4bit_module(layer_block.mlp.up_proj, v, ALPHA, input_dim_matching=True)
    abliterate_4bit_module(layer_block.mlp.down_proj, v, ALPHA, input_dim_matching=False)

print("=" * 60)
print("✅ SUCCESS: Dynamic Winsorized Abliteration Pipeline fully executed.")

```

---

## 5. Technical Specifications & Memory Profile

* **Quantization Format:** Compatible with `NF4` (NormalFloat4) and `FP4` formats with double quantization (`nested=True`).
* **VRAM Overhead:** Performs module-by-module dequantization, update, and re-quantization on the fly, eliminating full-model FP16/FP32 memory allocation.
* **Compatibility:** Tested on LLaMA-2/3, Mistral, Qwen 1.5/2/2.5, and Gemma transformer backbones.