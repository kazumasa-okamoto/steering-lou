from __future__ import annotations

import random
import re
from typing import Any

import torch

from lou_steering.data.chat import prompt_text


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def generate_once(
    model: Any,
    tokenizer: Any,
    prompt: str,
    seed: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> str:
    set_seed(seed)
    encoded = tokenizer(prompt_text(tokenizer, prompt), return_tensors="pt").to(model.device)
    input_len = encoded["input_ids"].shape[-1]
    with torch.inference_mode():
        out = model.generate(
            **encoded,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
    return strip_thinking(decoded)
