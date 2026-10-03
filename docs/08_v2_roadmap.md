# BitNarrow v2 Lab: Open Weight Model Surgery Program

Naming convention: canonical model names, HuggingFace repo ids, benchmark names, package names, and CLI flags keep their official spelling because they are identifiers. All other prose follows the no hyphen, no dash style.


# 1. Project Definition

## 1.1 Mission

Turn BitNarrow from one technique on one 1.5B model into a lab that takes any open weight base model from 1B up to the 40B class and transforms it along every axis a real AI team cares about: reasoning, knowledge, speed, size, context length, language, modality, tool use, architecture, and interpretability. Every transformation is measured by one shared evaluation spine, audited by ModelDiff, and shipped to HuggingFace in a form a developer can run in one command.

The story for a hiring panel: you did not train one model, you built the machinery that converts base models into specialized ones, measured every change, and shipped all of it as usable artifacts.

## 1.2 The umbrella design

Four shared pieces, many projects on top:

* `bitnarrow` package and CLI: one entry point, one recipe file per transformation, same command shape everywhere.
* Forge Eval spine: one harness, one results schema, one release gate. Nothing ships without it.
* ModelDiff hook: every artifact gets an automatic report of what changed against its base (weights, activations, behavior). This connects BitNarrow v2 to your flagship ModelDiff project so each strengthens the other.
* Showcase template: every HuggingFace repo looks and works the same way (section 1.7).

## 1.3 Compute tiers (assumed)

* Tier L, local: GTX 1650 4GB and GTX 1080Ti 11GB. Profiling, patch edits, eval and quantization for models up to about 3B at 4bit, all code development and smoke tests.
* Tier A, single 80GB GPU (H100 or A100 on RunPod): full fine tuning up to about 7B, LoRA and QLoRA up to 32B, inference and eval for 40B class models at 8bit or 4bit.
* Tier B, 8x H100 80GB node: full fine tuning up to 14B, LoRA up to 40B class with FSDP or DeepSpeed ZeRO 3, GRPO reinforcement learning at 7B to 32B, vision and audio adapter training, 70B teacher inference for distillation.

Rule: develop on Tier L, validate one recipe on Tier A at small size, only then spend Tier B hours.

## 1.4 Base model ladder

Verify the latest releases and licenses at Phase 0; newer open weights may supersede these candidates.

* S tier, 0.5B to 3B: Qwen2.5-1.5B-Instruct, Llama-3.2-3B-Instruct, gemma-2-2b-it
* M tier, 7B to 9B: Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, gemma-2-9b-it
* L tier, 14B: Qwen2.5-14B-Instruct, phi-4
* XL tier, 27B to 40B class: Qwen2.5-32B-Instruct, gemma-2-27b-it, Yi-1.5-34B-Chat
* MoE: Mixtral-8x7B-Instruct-v0.1 (about 47B total), Qwen3-30B-A3B
* Teachers (inference only): QwQ-32B, DeepSeek-R1-Distill-Qwen-32B, Llama-3.3-70B-Instruct
* Encoders for multimodal: siglip-so400m-patch14-384 (vision), whisper-large-v3 encoder (audio)

License check per base before publishing: Llama derivatives must carry "Llama" at the start of the model name and ship the Llama license; Gemma derivatives inherit Gemma terms; some Qwen sizes (for example 3B) use a research license, not Apache 2.0. The license field on every card is set from the base, never assumed.

## 1.5 Scope in

Fourteen projects, listed in section 2. Each is a separate showcase with its own HuggingFace collection, but all run on the shared umbrella.

## 1.6 Scope out and the non negotiable gate

* The refusal direction work (project Narrow) stays strictly an over refusal study: fewer refusals on safe prompts, harmful prompt refusal held at or above the base model. That is a hard release gate in Forge Eval applied to every artifact from every project, not only Narrow. A reasoning model, a merge, or a pruned model that loses harmful refusal does not ship either.
* No pretraining from scratch. Continued pretraining on bounded domain corpora is in scope.
* No custom CUDA or Triton kernels inside this program; that stays in the separate fused kernel project.

## 1.7 Showcase standard (every HuggingFace artifact)

Each model repo must have all of:

* Model card with a results table generated from the Forge Eval JSON (base vs artifact, every metric, gate status).
* ModelDiff report link.
* Three ways to run: transformers snippet, `vllm serve` command, GGUF for Ollama and llama.cpp.
* A Gradio Space demo (ZeroGPU where the model fits, quantized GGUF demo otherwise).
* The exact recipe file and the `bitnarrow` command that reproduces it.
* Known limitations section, honest about where the transformation hurt.

If a repo is missing any of these, it is not finished.


# 2. Project Catalogue

Each entry: purpose, technique, targets, published artifacts, success metric, tier.

## P0 Forge Eval (the spine)
* Purpose: one harness that scores any model on every axis and emits comparable JSON.
* Axes and benchmarks:
  * Reasoning: GSM8K, MATH-500, AIME 2024 and 2025, GPQA Diamond
  * Knowledge: MMLU-Pro, plus a held out domain QA set per Graft domain
  * Code: HumanEval+, MBPP+, LiveCodeBench
  * Instruction following: IFEval
  * Long context: RULER, needle in a haystack at 32K to 128K
  * Multimodal: MMMU, DocVQA, ChartQA; audio WER on LibriSpeech test clean
  * Tool use: BFCL
  * Indic: MILU, IndicGenBench
  * Over refusal: XSTest, OR-Bench; harmful refusal retained: HarmBench, StrongREJECT
  * Systems: time to first token, tokens per second, peak VRAM, disk size, speculative acceptance rate
* Artifacts: HuggingFace dataset repo with frozen splits, leaderboard Space, `bitnarrow eval` CLI.
* Success: reproduces published numbers of the unmodified bases within noise.
* Tier: L for code, A for XL eval runs.

## P1 Narrow (over refusal directional edit, v1 generalized)
* Technique: Winsorized activation profiling, Gram Schmidt projection across a discovered layer band, patch export.
* Targets: S, M, L, XL (projection is cheap; XL needs Tier A only for profiling memory).
* Artifacts: small weight patches per base, one collection.
* Success: over refusal down, harmful refusal not below base, capability within threshold.
* Tier: L for S, A for M to XL.

## P2 Think (instruct to reasoning conversion)
* Technique, three stages:
  * Stage 1 trace distillation: generate long chain of thought traces from QwQ-32B or DeepSeek-R1-Distill-Qwen-32B on math, code, and science prompts, filter by verified correctness, SFT the student.
  * Stage 2 GRPO with verifiable rewards: math answer checking and code unit tests as reward, format reward for thinking tags.
  * Stage 3 thinking budget control: train with explicit budget tokens so users can trade accuracy for latency.
* Targets: S (1.5B), M (7B), L (14B), XL (32B LoRA).
* Artifacts: `Think` models per size, the filtered trace dataset, the reward functions as reusable code, a Space showing the thinking trace live with a budget slider.
* Success: large AIME and MATH-500 gains over the instruct base, accuracy vs budget curve published.
* Tier: A for S and M SFT, B for GRPO and XL.

## P3 Graft (knowledge injection)
* Technique comparison, which is the real research value:
  * Continued pretraining on a bounded domain corpus, then short SFT to restore chat behavior.
  * LoRA SFT on synthetic domain QA generated from the corpus.
  * Targeted fact editing with MEMIT for small sets of facts.
  * RAG baseline with the untouched model, so you can show when weights beat retrieval and when they do not.
* Domains: pick two with public corpora, for example medical (connects to your clinical work) and Indian law or finance.
* Targets: M and L, one XL for the stronger domain.
* Artifacts: domain models, the curated corpora, a comparison report of the four approaches.
* Success: domain QA up, MMLU-Pro general retention within threshold (catastrophic forgetting measured, not hidden).
* Tier: A, B for continued pretraining at L and XL.

## P4 Velocity (fast response)
* Technique:
  * Draft models for speculative decoding: distill a 0.5B to 1.5B draft that shares the tokenizer with each M, L, XL target.
  * EAGLE style draft heads trained on target hidden states.
  * KV cache quantization and FP8 serving configs.
* Targets: drafts paired with every M, L, XL artifact, including Think models.
* Artifacts: draft models and heads, ready `vllm serve` configs, a benchmark Space showing speedup.
* Success: measured end to end speedup with identical outputs (speculative decoding is lossless when verified), acceptance rate reported per target.
* Tier: A, B for XL target hidden state collection.

## P5 Shrink (smaller with retained capability)
* Technique: depth pruning by block importance (angular distance), width pruning of MLP channels and attention heads by activation importance, then knowledge distillation from the unpruned model (logit KL plus hidden state loss).
* Targets: 32B to about 20B, 14B to about 9B, 8B to about 4B.
* Artifacts: pruned and healed models with compression curves.
* Success: healed model beats a same size off the shelf model on Forge Eval.
* Tier: B for distillation at L and XL.

## P6 Grow (bigger and new architecture)
* Technique:
  * Depth upscaling: duplicate and interleave layers, then heal with continued training.
  * Dense to MoE upcycling: copy the MLP into N experts, add a router, continue training so experts specialize.
  * MoE expert pruning and merging on Mixtral-8x7B and Qwen3-30B-A3B to cut memory.
* Targets: upscale M to about 11B, upcycle S and M to MoE, prune XL MoE experts.
* Artifacts: upscaled and upcycled models, expert routing analysis visualized in a Space.
* Success: upscaled model beats its base; upcycled MoE beats the dense base at equal active parameters.
* Tier: B.

## P7 Blend (merging)
* Technique: SLERP, TIES, DARE via mergekit, plus evolutionary merge search where Forge Eval is the fitness function.
* Targets: merge your own artifacts (for example Think plus Graft at the same base) to show composition.
* Artifacts: merged models with parent comparison tables, the search code.
* Success: merge beats both parents on the combined objective without regressing the gate.
* Tier: L and A (merging is cheap; eval is the cost).

## P8 Stretch (long context)
* Technique: YaRN RoPE scaling, short continued training on long documents, long context SFT.
* Targets: M and L from 32K to 128K.
* Artifacts: extended models, RULER heatmaps on the card.
* Success: RULER at 128K above a defined floor, short context scores unchanged.
* Tier: B (long sequences need memory).

## P9 Lipi (Indic language extension)
* Technique: extend the tokenizer with Kannada, Hindi, Tamil, Telugu tokens, initialize new embeddings from subword averages, continued pretraining on Indic corpora, bilingual SFT.
* Targets: M and L.
* Artifacts: extended tokenizer, models, token efficiency report (tokens per sentence before vs after).
* Success: MILU and IndicGenBench gains, English retained, fewer tokens per Indic sentence so inference is cheaper.
* Tier: B.

## P10 Eyes and Ears (text to multimodal conversion)
* Technique:
  * Vision: SigLIP encoder plus MLP projector, stage 1 projector alignment on image caption pairs, stage 2 visual instruction tuning (LLaVA recipe).
  * Audio: Whisper encoder plus projector, speech instruction tuning for spoken input.
* Targets: M and L for vision, M for audio, one XL vision flagship.
* Artifacts: vision language and speech language models, Spaces where users upload an image or speak.
* Success: MMMU, DocVQA, ChartQA, WER competitive with same size open multimodal models; text scores retained.
* Tier: B.

## P11 Act (tool use and structured output)
* Technique: SFT on function calling traces with JSON schema, constrained decoding config, multi turn agent trajectories.
* Targets: S and M (small fast agents are what developers deploy).
* Artifacts: tool calling models, an OpenAI compatible serving example.
* Success: BFCL gains, valid JSON rate near 100 percent.
* Tier: A.

## P12 Lens (interpretability)
* Technique: train sparse autoencoders on residual stream activations at chosen layers, release feature dictionaries, build steering vectors from features, compare against the Narrow direction.
* Targets: S and M, one layer set on XL.
* Artifacts: SAE weights, a feature browser Space, a steering library.
* Success: reconstruction loss and downstream loss delta reported; at least one feature shown to causally steer behavior.
* Tier: A, B for XL activation collection.

## P13 Pack (deployment matrix)
* Technique: export every surviving artifact to GGUF (several quant levels), AWQ, GPTQ, EXL2, FP8, bitsandbytes 4bit.
* Artifacts: quant repos linked from each main card, size and speed table.
* Success: every artifact runnable on a laptop via GGUF and on a server via vLLM.
* Tier: L for S and M, A for XL.

## P14 Flagship (the composition showcase)
* One XL model that stacks the program: 32B base, Think reasoning, Graft domain knowledge, Eyes vision, Velocity draft model, Pack exports, full ModelDiff audit.
* This is the single link you put at the top of your resume.
* Tier: B.


# 3. Phased Architecture

* Phase 0: Foundation. Package, Forge Eval, ModelDiff hook, showcase template, training infra configs.
* Phase 1: Local proof. Narrow, Blend, Pack on S tier. Proves the full loop end to end on your own GPUs.
* Phase 2: Reasoning. Think from S up to XL.
* Phase 3: Speed and size. Velocity and Shrink.
* Phase 4: Knowledge, context, language. Graft, Stretch, Lipi.
* Phase 5: Modality and tools. Eyes and Ears, Act.
* Phase 6: Architecture growth. Grow (upscale, MoE upcycling, expert pruning).
* Phase 7: Interpretability. Lens.
* Phase 8: Flagship. P14 composition at XL.
* Phase 9: Launch. Collections, leaderboard, writeups, release tag.

Phases 2 through 7 are mostly independent after Phase 1, so they can be reordered by interview priority or GPU budget. Phase 8 depends on 2, 3, 4, 5.


# 4. Task Execution Matrix

## Phase 0: Foundation

Tasks:
* Create the package layout (section 6), the recipe file format (YAML: base, data, method, hyperparameters, eval set, output repo id), and the CLI verbs `run`, `eval`, `diff`, `export`, `publish`.
* Build Forge Eval on top of lm-evaluation-harness and lighteval for text, custom runners for safety, speed, multimodal, and tool use. Freeze splits, publish as a dataset repo.
* Define the results schema (versioned JSON) and the release gate: harmful refusal at or above base, general capability drop within a fixed threshold, axis specific success metric met.
* Wire ModelDiff so `bitnarrow diff base artifact` produces a report saved next to the eval JSON.
* Write the model card generator and a Gradio Space template.
* Training infra: Accelerate plus FSDP and DeepSpeed ZeRO 3 configs, TRL for SFT and GRPO, vLLM for generation inside GRPO and teacher sampling, checkpoint sync to HuggingFace or object storage so RunPod pods are disposable.
* Run Forge Eval on every base in the ladder to create the baseline table.

Dependencies: transformers, trl, peft, accelerate, deepspeed, vllm, lm-evaluation-harness, lighteval, mergekit, llama.cpp, autoawq, gptqmodel, gradio.

Goal: baselines published; any model can be scored with one command.

## Phase 1: Local proof

Tasks:
* Port v1 into the Narrow module, config driven layer band discovery, run on S tier bases.
* Two Blend merges on S tier.
* Pack exports for every passing S artifact.
* Push through the full showcase standard: card, Space, ModelDiff, three run methods.

Goal: three complete, polished repos built only on local GPUs. This validates the template before any paid compute.

## Phase 2: Reasoning (Think)

Tasks:
* Build the prompt pool (math, code with tests, science), sample teacher traces with vLLM on Tier B, verify and filter, publish the trace dataset.
* SFT S and M, eval, then GRPO with verifiable rewards on M.
* Add budget tokens, train, measure accuracy vs budget.
* Scale to L full or LoRA, XL LoRA.

Dependencies: Phase 0 infra, Tier B hours.

Goal: a Think family across four sizes with a published reasoning curve.

## Phase 3: Speed and size

Tasks:
* Velocity: distill a matching draft per target family, train EAGLE style heads for the Think XL model, benchmark in vLLM.
* Shrink: importance scoring, prune, distill heal, compare against same size off the shelf models.

Goal: every major artifact has a paired draft model; at least two compressed models beat same size competitors.

## Phase 4: Knowledge, context, language

Tasks:
* Graft: build two domain corpora and held out QA, run the four way comparison, publish the report.
* Stretch: YaRN extension plus long context training on M and L, RULER heatmaps.
* Lipi: tokenizer extension, embedding init, continued pretraining, bilingual SFT, token efficiency report.

Goal: three distinct capability conversions, each with a forgetting analysis.

## Phase 5: Modality and tools

Tasks:
* Eyes: projector alignment, visual instruction tuning on M and L, then XL.
* Ears: Whisper encoder projector training on M.
* Act: function calling SFT on S and M, OpenAI compatible serving example.

Goal: working image and speech input demos in Spaces; small agent models with BFCL numbers.

## Phase 6: Architecture growth

Tasks:
* Depth upscale M, heal, compare.
* Upcycle S and M dense to MoE, continued training, routing analysis.
* Expert pruning on Mixtral-8x7B and Qwen3-30B-A3B.

Goal: architecture changes shown to improve compute vs quality tradeoffs, with routing visualizations.

## Phase 7: Interpretability

Tasks:
* Collect activations, train SAEs on S and M layers, publish dictionaries.
* Feature browser Space, steering library, comparison with the Narrow direction.

Goal: an interpretability artifact that links back to the mechanism behind P1.

## Phase 8: Flagship

Tasks:
* Start from Think XL, merge or continue train with Graft domain data, attach Eyes vision adapter, pair the Velocity draft, export with Pack.
* Full Forge Eval, full ModelDiff report, launch Space.

Goal: one model page that demonstrates the entire program.

## Phase 9: Launch

Tasks:
* One collection per project plus a master collection.
* Leaderboard Space reading every result JSON from a manifest.
* Technical writeups in `docs` per project, including negative results.
* GitHub release v2, PyPI publish of `bitnarrow`.
* Short demo video of the CLI converting a base model end to end.

Goal: everything public, reproducible, and linkable.


# 5. Compute Budget (planning estimates, recompute after Phase 1)

Rough H100 GPU hours, order of magnitude only:

* Phase 0 baselines across the ladder: about 50 to 100
* Phase 1: near zero paid (local)
* Phase 2 Think across four sizes including GRPO: about 800 to 1500
* Phase 3: about 300 to 600
* Phase 4: about 800 to 1500 (continued pretraining dominates)
* Phase 5: about 600 to 1200
* Phase 6: about 500 to 1000
* Phase 7: about 100 to 300
* Phase 8: about 300 to 600

Cost control rules: spot or community pods, checkpoint every N steps to remote storage, never run an XL job before its S version passes the gate, cut XL targets first if budget runs short.


# 6. Repository and Package Layout

```
bitnarrow/
  cli.py
  recipe.py
  eval/
    harness.py
    gate.py
    schema.py
    runners/
  surgery/
    narrow.py
    prune.py
    upscale.py
    upcycle.py
    merge.py
    longctx.py
    tokenizer_ext.py
  train/
    sft.py
    grpo.py
    distill.py
    cpt.py
  multimodal/
    vision.py
    audio.py
  speed/
    draft.py
    eagle.py
  interp/
    sae.py
    steer.py
  export/
    quant.py
    card.py
    space.py
  diff/
    modeldiff_hook.py
recipes/
  narrow/
  think/
  graft/
  velocity/
  shrink/
  grow/
  blend/
  stretch/
  lipi/
  eyes/
  act/
  lens/
  flagship/
infra/
  accelerate/
  deepspeed/
  runpod/
spaces/
  leaderboard/
  template/
docs/
```

Developer experience target:

```
pip install bitnarrow
bitnarrow run recipes/think/qwen2.5-7b.yaml
bitnarrow eval AmareshHebbar/Qwen2.5-7B-Think
bitnarrow diff Qwen/Qwen2.5-7B-Instruct AmareshHebbar/Qwen2.5-7B-Think
bitnarrow publish AmareshHebbar/Qwen2.5-7B-Think
```

Repo id pattern: `AmareshHebbar/{Base}-{Project}` with Llama derivatives starting with "Llama" per license.


# 7. Path to Completion

## How to start
Phase 0 only. The first milestone is the baseline table: Forge Eval reproducing published numbers for every base. If baselines do not match, the harness is wrong and nothing else is trustworthy.

## How to test
Every artifact: one `bitnarrow eval` run, one `bitnarrow diff` run, gate check. Pass publishes, fail logs a negative result into the project writeup. Spaces are smoke tested on every publish.

## How to bridge milestones
* A project scales up a tier only when its smaller tier artifact has passed the gate.
* Reusable stages are built once: distillation code from Think is reused in Shrink, Velocity, and Graft; continued pretraining from Graft is reused in Lipi and Stretch.
* Schema changes are versioned bumps, never silent edits.

## How it concludes
Done means: PyPI package live, leaderboard live, every project has at least one passing artifact at M tier or above with the full showcase standard, the Flagship is published, all writeups are in docs, and the v2 tag is cut.


# 8. Interview Narrative

Order to present it:
1. The problem: base models need to be converted for real products, and conversions break things silently.
2. Forge Eval and the release gate: you measure every axis, including safety retention, before shipping.
3. ModelDiff integration: you can show exactly what each conversion changed inside the model.
4. Breadth: reasoning, knowledge, speed, size, context, language, modality, tools, architecture.
5. Scale: same machinery from 1.5B to the 40B class on multi GPU.
6. Honesty: negative results and forgetting analyses published, not hidden.
7. Usability: every result is one command away for any developer.

Pick the two or three projects closest to the company you are interviewing with and lead with those.


# 9. Open Decisions

* Real GPU budget in H100 hours or currency per month. This decides whether XL targets are in Phases 2 to 7 or only in the Flagship.
* Gate threshold: the exact allowed general capability drop.
* Graft domains: which two.
* Lipi languages: all four or Kannada and Hindi first.
* Priority order of Phases 2 to 7 based on which roles you are targeting next.
