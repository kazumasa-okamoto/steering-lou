from __future__ import annotations

from typing import Any


def hiragana_count(text: str) -> int:
    return sum(1 for ch in text if "\u3040" <= ch <= "\u309f")


def kanji_count(text: str) -> int:
    return sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")


def japanese_count(text: str) -> int:
    return hiragana_count(text) + kanji_count(text)


def katakana_count(text: str) -> int:
    return sum(1 for ch in text if "\u30a0" <= ch <= "\u30ff")


def english_count(text: str) -> int:
    return sum(1 for ch in text if ("A" <= ch <= "Z") or ("a" <= ch <= "z"))


def score_output(text: str) -> dict[str, Any]:
    japanese_chars = japanese_count(text)
    katakana_chars = katakana_count(text)
    english_chars = english_count(text)
    denominator = japanese_chars + katakana_chars + english_chars

    if denominator == 0:
        japanese_rate = 0.0
        katakana_rate = 0.0
        english_rate = 0.0
    else:
        japanese_rate = japanese_chars / denominator
        katakana_rate = katakana_chars / denominator
        english_rate = english_chars / denominator

    return {
        "metric_denominator": denominator,
        "japanese_chars": japanese_chars,
        "katakana_chars": katakana_chars,
        "english_chars": english_chars,
        "japanese_rate": japanese_rate,
        "katakana_rate": katakana_rate,
        "english_rate": english_rate,
    }
