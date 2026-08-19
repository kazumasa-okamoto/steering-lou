from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from constants import TRIPLET_PATH

REQUIRED_TRIPLET_KEYS = {"ja", "lou", "en"}


def load_triplets(limit: int | None = None, path: Path = TRIPLET_PATH) -> list[dict[str, str]]:
    triplets = json.loads(path.read_text(encoding="utf-8"))
    validate_triplets(triplets, path)
    return triplets[:limit] if limit else triplets


def validate_triplets(triplets: Any, path: Path) -> None:
    if not isinstance(triplets, list):
        raise ValueError(f"{path} must contain a JSON array")

    for idx, triplet in enumerate(triplets):
        if not isinstance(triplet, dict):
            raise ValueError(f"Triplet {idx} in {path} must be an object")

        missing = REQUIRED_TRIPLET_KEYS - triplet.keys()
        if missing:
            raise ValueError(f"Triplet {idx} in {path} is missing keys: {sorted(missing)}")

        for key in REQUIRED_TRIPLET_KEYS:
            if not isinstance(triplet[key], str) or not triplet[key].strip():
                raise ValueError(f"Triplet {idx} key {key!r} in {path} must be a non-empty string")


# Backward-compatible alias for older imports.
load_pairs = load_triplets
