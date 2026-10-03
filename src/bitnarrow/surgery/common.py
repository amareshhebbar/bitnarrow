from __future__ import annotations

from bitnarrow.models import format_prompts, model_device


def collect_last_token_hidden(model, tokenizer, prompts: list[str], batch_size: int = 8, chat: bool = True):
    import torch

    if not prompts:
        raise ValueError("no prompts supplied for activation collection")
    texts = format_prompts(tokenizer, prompts, chat)
    device = model_device(model)
    chunks = []
    for i in range(0, len(texts), batch_size):
        batch = tokenizer(texts[i : i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            out = model(**batch, output_hidden_states=True, use_cache=False)
        last = torch.stack([h[:, -1, :] for h in out.hidden_states], dim=0)
        chunks.append(last.float().cpu())
    return torch.cat(chunks, dim=1)
