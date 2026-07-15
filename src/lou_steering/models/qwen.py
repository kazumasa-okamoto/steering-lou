from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from steering_vectors import guess_and_enhance_layer_config
from steering_vectors.layer_matching import collect_matching_layers
from transformers import AutoModelForCausalLM, AutoTokenizer

from lou_steering.constants import LAYER_TYPE, MODEL_ID, OUT_DIR


def load_model_and_tokenizer(model_id: str = MODEL_ID) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


def get_layer_info(
    model: Any, layer_type: str = LAYER_TYPE, out_dir: Path = OUT_DIR
) -> tuple[dict[str, Any], list[str]]:
    layer_config = guess_and_enhance_layer_config(model, layer_type=layer_type)
    matcher = layer_config[layer_type]
    matching_layers = collect_matching_layers(model, matcher)
    info = {
        "layer_type": layer_type,
        "matcher": matcher,
        "num_layers": len(matching_layers),
        "matching_layers": matching_layers,
        "note": "steering-vectors records and patches the forward output of each matched decoder block; for Qwen3 this is the block output residual stream.",
    }
    (out_dir / "layer_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return info, matching_layers
