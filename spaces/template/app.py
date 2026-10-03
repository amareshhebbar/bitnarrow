import os
from threading import Thread

import gradio as gr
import torch
from transformers import TextIteratorStreamer

import bitnarrow

REPO = os.environ.get("BITNARROW_REPO", "AmareshHebbar/Qwen2.5-1.5B-Narrow")
BASE = os.environ.get("BITNARROW_BASE")

model, tokenizer = bitnarrow.load(REPO, dtype="bfloat16" if torch.cuda.is_available() else "float32")
base_model = None
if BASE:
    from bitnarrow.models import load_model

    base_model = load_model(BASE, dtype="bfloat16" if torch.cuda.is_available() else "float32")


def _stream(m, history, message, max_new_tokens, temperature):
    msgs = [{"role": h["role"], "content": h["content"]} for h in history] + [{"role": "user", "content": message}]
    ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(m.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    kwargs = {
        "input_ids": ids,
        "streamer": streamer,
        "max_new_tokens": int(max_new_tokens),
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if temperature > 0:
        kwargs["temperature"] = float(temperature)
    Thread(target=m.generate, kwargs=kwargs).start()
    text = ""
    for chunk in streamer:
        text += chunk
        yield text


def chat(message, history, max_new_tokens, temperature):
    yield from _stream(model, history, message, max_new_tokens, temperature)


def compare(message, max_new_tokens):
    left = "".join(list(_stream(base_model, [], message, max_new_tokens, 0.0))[-1:]) if base_model else "set BITNARROW_BASE to enable"
    right = "".join(list(_stream(model, [], message, max_new_tokens, 0.0))[-1:])
    return left, right


with gr.Blocks(title=REPO.split("/")[-1]) as demo:
    gr.Markdown(f"# {REPO.split('/')[-1]}\nEdited with [bitnarrow](https://github.com/amareshhebbar/bitnarrow). Model card: [{REPO}](https://huggingface.co/{REPO})")
    with gr.Tab("Chat"):
        tokens = gr.Slider(16, 1024, value=256, step=16, label="max new tokens")
        temp = gr.Slider(0.0, 1.5, value=0.0, step=0.05, label="temperature")
        gr.ChatInterface(chat, type="messages", additional_inputs=[tokens, temp])
    if base_model is not None:
        with gr.Tab("Base vs edited"):
            prompt = gr.Textbox(label="prompt", value="How do I kill a Python process that hangs?")
            n = gr.Slider(16, 512, value=128, step=16, label="max new tokens")
            go = gr.Button("Compare")
            with gr.Row():
                out_base = gr.Textbox(label="base", lines=12)
                out_edit = gr.Textbox(label="edited", lines=12)
            go.click(compare, [prompt, n], [out_base, out_edit])

if __name__ == "__main__":
    demo.queue().launch()
