# 🎯 Part 1: Activation Hunting & Dimension Mismatches

## The Game Plan
ididn't want to mess around with system prompts or fragile jailbreaks. The goal was simple: treat safety alignment as a straight line (a vector) inside the model's high-dimensional brain and erase it. 

To do that, ihad to catch the model in the act of preparing a refusal. iset up a diagnostic run feeding the model two distinct sets of prompts via Unsloth:
- **Harmless Prompts:** Simple questions like baking a cookie or writing a story about a frog.
- **Harmful Prompts:** Unsafe queries like hotwiring a car or bypassing standard Wi-Fi handshakes.

iset up a PyTorch forward hook to wiretap the network at the exact moment it evaluates the input prompt and prepares to predict its **very first output token**—because that is precisely where the model "decides" whether to cooperate or panic.

---

## The Mistakes iMade (and How iFixed Them)

### Bug #1: The Function Exit Trait (Indentation Hell)
When ifirst wrote the tokenization loop function `format_tokenize()`, our print statements kept telling us `SO built... 1 harmless inputs and 1 harmful inputs`. Out of an array of 5 prompts, it only processed the first one.
* **What went wrong:** Classic indentation typo. The `return token_inputs` statement was accidentally pushed inside the `for` loop body, forcing the function to exit instantly after checking the very first row.
* **The Fix:** Shifted the `return` statement back four spaces outside the block to let the loop populate the entire array.

### Bug #2: The Hidden Tuple Trap
Once the dataset was fixed, the forward hook immediately threw a nasty `TypeError: tuple indices must be integers or slices, not tuple` right inside the tensor extraction logic.
* **What went wrong:** Inside PyTorch forward hooks, the `input` argument is passed as a standard Python **tuple**, not a clean tensor block. Writing `hidden_states[0, -1, :]` meant iwere trying to slice a tuple across multiple dimensions, which Python hates.
* **The Fix:** ihad to unpack the first index of the tuple explicitly (`hidden_tensor = input_tuple[0]`) to get the actual multi-dimensional tensor out before attempting scalar slicing.

### Bug #3: The 8960 vs 1536 Shape Explosion
This was our first major mathematical wall. ioriginally hooked into the middle section of the model's factual database (`mlp.down_proj`) and extracted an array of vectors. But when itried to apply those vectors to neutralize the attention layers (`self_attn.o_proj`), the script crashed with:
`RuntimeError: mat1 and mat2 shapes cannot be multiplied (1536x1536 and 8960x1)`
* **What went wrong:** In Qwen architectures, the inner MLP layers temporarily expand the model's dimensions up to **8960** features to calculate complex relations, before projecting them back down to the model's standard hidden size of **1536**. Our vectors were 8960 dimensions long, making them completely incompatible with the attention modules.
* **The Fix:** ishifted our wiretap location to the input gateway of the MLP (`mlp.gate_proj`). This layer taps directly into the standard 1536-dimensional hidden highway, giving us pristine vector shapes that match both attention and feed-forward dimensions perfectly.

### Bug #4: The CUDA Stream Ghost (Advanced Nightmare)
*This was a brutal, hard-to-trace asynchronous runtime issue.* When running bulk inference, some layer keys in our storage dictionary were showing completely empty or returning raw zero arrays randomly.
* **What went wrong:** Because PyTorch runs GPU operations asynchronously, the model was sprinting ahead executing later layers while our `.cpu()` copy command was still trying to read data from the GPU. The hook was grabbing memory that hadn't finished calculating yet.
* **The Fix:** Injected an explicit `torch.cuda.synchronize()` right before detaching the tensors to force the GPU threads to pause and finish writing their hidden values before icopied them over to our tracking arrays.

---

## The Matrix Geometry Map
Once the keys and shapes were aligned, icalculated the average vectors of our two camps and ran a **Cosine Similarity Check** layer by layer. The output was clear:

```text
Layer 0   : 0.9972 (Identical behavior - just parsing words)
Layer 10  : 0.9790 
Layer 14  : 0.9344 (The divergence starts here)
Layer 16  : 0.8813 ──► [TARGET STARTS HERE]
Layer 22  : 0.7673 ──► [THE REFUSAL ZONE - HIGH GEOMETRIC SPLIT]
Layer 24  : 0.7076 ──► [THE ABSOLUTE CENSORSHIP PEAK]
Layer 27  : 0.8564 (Formatting output response text)
```

The data proved it: the model is not censored in the beginning or end. The entire safety engine executes as a massive geometric wall between **Layers 16 and 27**.

---

## Next Steps
Now that imapped the precise coordinates of the refusal zone, iwere ready to calculate the vectors and execute the weight projections. But the model had another trick up its sleeve: **Massive Activation Spikes**. 

Move on to **[Part 2: The Winsorization Breakthrough](./02_winsor_breakthrough.md)** to see how the model used numeric outliers to bypass our first ablation tests.
