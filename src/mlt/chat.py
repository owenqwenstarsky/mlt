"""`mlt chat`: a minimal REPL to sanity-check a trained model or adapters."""

from __future__ import annotations

import json
from pathlib import Path

from .config.schema import MethodConfig
from .models.loading import is_adapter_dir, load_pretrained


def load_for_chat(path: Path):
    from mlx_lm.utils import load_adapters, load_model, load_tokenizer

    if is_adapter_dir(path):
        from .export import adapter_base

        meta = json.loads((path / "mlt_adapter.json").read_text())
        method = MethodConfig(**{k: v for k, v in meta["method"].items() if k != "base_path"})
        base = adapter_base(path)
        # Re-create the base exactly as trained (quantized for QLoRA), then attach adapters.
        lm = load_pretrained(base, method, "bf16")
        model = load_adapters(lm.model, str(path))
        config = lm.hf_config
    else:
        model, config = load_model(path)
    model.eval()
    eos = config.get("eos_token_id")
    tokenizer = load_tokenizer(path, eos_token_ids=eos if isinstance(eos, list) else ([eos] if eos is not None else None))
    return model, tokenizer


def repl(path: Path, max_tokens: int, system: str | None) -> None:
    from mlx_lm import stream_generate

    model, tokenizer = load_for_chat(path)
    messages = [{"role": "system", "content": system}] if system else []
    print("type a message (/reset to clear, Ctrl-D to quit)")
    while True:
        try:
            user = input("> ").strip()
        except EOFError:
            print()
            return
        if not user:
            continue
        if user == "/reset":
            messages = messages[:1] if system else []
            continue
        messages.append({"role": "user", "content": user})
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        reply = ""
        for chunk in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens):
            print(chunk.text, end="", flush=True)
            reply += chunk.text
        print()
        messages.append({"role": "assistant", "content": reply})
