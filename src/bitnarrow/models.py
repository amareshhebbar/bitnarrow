from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PATCH_WEIGHTS = "patch.safetensors"
PATCH_META = "patch.json"


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("install bitnarrow[torch] for model operations") from exc
    return torch


def resolve_dtype(name: str | None):
    torch = _torch()
    if name in (None, "auto"):
        return "auto"
    table = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if name not in table:
        raise ValueError(f"unknown dtype {name!r}")
    dtype = table[name]
    if dtype is torch.bfloat16 and not torch.cuda.is_available():
        return torch.float32
    return dtype


def _dtype_kwarg() -> str:
    import transformers

    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    return "dtype" if (major, minor) >= (4, 56) else "torch_dtype"


def load_tokenizer(model_id: str, revision: str | None = None, trust_remote_code: bool = False):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.model_input_names = ["input_ids", "attention_mask"]
    return tok


def load_model(
    model_id: str,
    revision: str | None = None,
    dtype: str | None = "bfloat16",
    load_in_4bit: bool = False,
    trust_remote_code: bool = False,
    patch: str | Path | None = None,
    device_map: str | dict | None = "auto",
):
    torch = _torch()
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {
        "revision": revision,
        "trust_remote_code": trust_remote_code,
        _dtype_kwarg(): resolve_dtype(dtype),
    }
    if device_map is not None and torch.cuda.is_available():
        kwargs["device_map"] = device_map
    patch_tensors = read_patch(patch)[0] if patch else None
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        skip = ["lm_head"]
        if patch_tensors:
            skip += sorted({n.rsplit(".", 1)[0] for n in patch_tensors})
        compute = resolve_dtype(dtype)
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute if compute != "auto" else torch.bfloat16,
            llm_int8_skip_modules=skip,
        )
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    if patch_tensors:
        apply_patch(model, patch_tensors)
    return model


def read_patch(patch: str | Path | dict) -> tuple[dict[str, Any], dict[str, Any]]:
    if isinstance(patch, dict):
        return patch, {}
    from safetensors.torch import load_file

    path = Path(patch)
    if path.is_dir():
        weights, meta_path = path / PATCH_WEIGHTS, path / PATCH_META
    else:
        weights, meta_path = path, path.with_name(PATCH_META)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return load_file(str(weights)), meta


def write_patch(tensors: dict[str, Any], meta: dict[str, Any], out_dir: str | Path) -> Path:
    from safetensors.torch import save_file

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    clean = {k: v.detach().to("cpu").contiguous() for k, v in tensors.items()}
    save_file(clean, str(out / PATCH_WEIGHTS), metadata={"format": "pt", "producer": "bitnarrow"})
    (out / PATCH_META).write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return out


def apply_patch(model, patch: str | Path | dict) -> list[str]:
    torch = _torch()
    tensors = read_patch(patch)[0] if not isinstance(patch, dict) else patch
    applied = []
    with torch.no_grad():
        for name, value in tensors.items():
            param = model.get_parameter(name)
            if type(param).__name__ == "Params4bit":
                raise RuntimeError(
                    f"{name} is 4bit quantized; load with bitnarrow.load(..., patch=...) so patched modules stay unquantized"
                )
            if param.shape != value.shape:
                raise ValueError(f"shape mismatch for {name}: model {tuple(param.shape)} vs patch {tuple(value.shape)}")
            param.copy_(value.to(device=param.device, dtype=param.dtype))
            applied.append(name)
    return applied


def _fetch(ref: str) -> Path:
    path = Path(ref)
    if path.exists():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(ref, allow_patterns=[PATCH_WEIGHTS, PATCH_META, "*.json", "*.safetensors", "*.model", "*.txt"]))


def load(
    ref: str,
    patch: str | None = None,
    dtype: str | None = "bfloat16",
    load_in_4bit: bool = False,
    revision: str | None = None,
    trust_remote_code: bool = False,
):
    base_id = ref
    if patch is None:
        local = _fetch(ref)
        if (local / PATCH_META).exists():
            meta = json.loads((local / PATCH_META).read_text(encoding="utf-8"))
            patch, base_id = str(local), meta["base_model"]
            revision = revision or meta.get("base_revision")
        elif (local / "config.json").exists():
            base_id = str(local)
    elif not Path(patch).exists():
        patch = str(_fetch(patch))
    model = load_model(
        base_id,
        revision=revision,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        trust_remote_code=trust_remote_code,
        patch=patch,
    )
    tok = load_tokenizer(base_id, revision=revision, trust_remote_code=trust_remote_code)
    return model, tok


def format_prompts(tokenizer, prompts: list[str], chat: bool = True) -> list[str]:
    if chat and getattr(tokenizer, "chat_template", None):
        return [
            tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
            for p in prompts
        ]
    return list(prompts)


def model_device(model):
    return next(model.parameters()).device


def generate(model, tokenizer, prompts: list[str], max_new_tokens: int = 64, batch_size: int = 8, chat: bool = True):
    torch = _torch()
    texts = format_prompts(tokenizer, prompts, chat)
    device = model_device(model)
    outputs: list[str] = []
    for i in range(0, len(texts), batch_size):
        batch = tokenizer(texts[i : i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            out = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new = out[:, batch["input_ids"].shape[1] :]
        outputs.extend(tokenizer.batch_decode(new, skip_special_tokens=True))
    return outputs


def decoder_layers(model):
    for path in ("model.layers", "model.decoder.layers", "transformer.h", "gpt_neox.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj, path
        except AttributeError:
            continue
    raise ValueError(f"cannot locate decoder layers on {type(model).__name__}")
