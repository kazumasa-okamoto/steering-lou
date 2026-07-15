from __future__ import annotations

from typing import Any


def apply_chat_template(
    tokenizer: Any, messages: list[dict[str, str]], add_generation_prompt: bool
) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def representation_messages(sentence: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": sentence}]


def representation_text(tokenizer: Any, sentence: str) -> str:
    return apply_chat_template(
        tokenizer, representation_messages(sentence), add_generation_prompt=True
    )


def prompt_text(tokenizer: Any, prompt: str) -> str:
    return apply_chat_template(
        tokenizer,
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
    )
