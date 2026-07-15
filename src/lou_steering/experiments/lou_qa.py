from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import asdict
from typing import Any

import torch
from steering_vectors import SteeringVector, guess_and_enhance_layer_config

from lou_steering.config import RunConfig, choose_layers, mode_settings
from lou_steering.constants import GENERATION_PROMPTS, LAYER_TYPE, MODEL_ID, OUT_DIR
from lou_steering.data.pairs import load_triplets
from lou_steering.evaluation.metrics import score_output
from lou_steering.evaluation.reporting import (
    save_cosine_summary,
    save_projection_summary,
    summarize_results,
)
from lou_steering.generation.sampling import generate_once
from lou_steering.models.qwen import get_layer_info, load_model_and_tokenizer
from lou_steering.runtime.environment import save_environment
from lou_steering.steering.triplet_vectors import (
    TripletSteeringVectors,
    cosine_similarity_by_layer,
    projection_check_by_layer,
    save_triplet_vectors,
    train_triplet_steering_vectors,
)

POOLING_METHODS = ["mean", "final"]


def run(mode: str) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    env = save_environment()
    print(json.dumps({k: env[k] for k in env if k != "nvidia_smi"}, ensure_ascii=False, indent=2))

    model, tokenizer = load_model_and_tokenizer()
    layer_info, _ = get_layer_info(model)
    num_layers = layer_info["num_layers"]
    all_layers = list(range(num_layers))
    selected_layers = choose_layers(num_layers)
    settings = mode_settings(mode, selected_layers)

    triplets = load_triplets(settings.triplet_limit)
    prompts = GENERATION_PROMPTS[: settings.prompt_count]
    seed = 20260715
    temperature = 0.7
    top_p = 0.9

    layer_config = guess_and_enhance_layer_config(model, layer_type=LAYER_TYPE)
    vectors = train_triplet_steering_vectors(
        model, tokenizer, triplets, layers=all_layers, layer_config=layer_config
    )
    vectors = move_vectors_to_device(vectors, model.device, torch.bfloat16)
    save_triplet_vectors(vectors, OUT_DIR, mode)

    cosine_by_pooling = {
        pooling: cosine_similarity_by_layer(
            vectors.lou_minus_ja[pooling], vectors.en_minus_ja[pooling]
        )
        for pooling in POOLING_METHODS
    }
    save_cosine_summary(cosine_by_pooling, mode)

    projection_by_pooling = {
        pooling: projection_check_by_layer(
            vectors.lou_perp_en[pooling], vectors.en_minus_ja[pooling]
        )
        for pooling in POOLING_METHODS
    }
    save_projection_summary(projection_by_pooling, mode)

    config = RunConfig(
        mode=mode,
        model_id=MODEL_ID,
        layers=all_layers,
        steering_layers=settings.steering_layers,
        coefficients=settings.coefficients,
        triplet_count=len(triplets),
        prompt_count=len(prompts),
        seed=seed,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=settings.max_new_tokens,
        layer_type=LAYER_TYPE,
        residual_matcher=str(layer_info["matcher"]),
        pooling_methods=POOLING_METHODS,
    )
    (OUT_DIR / f"run_config_{mode}.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    json_path = OUT_DIR / f"generations_{mode}.json"
    generation_records = generate_steering_comparison(
        model=model,
        tokenizer=tokenizer,
        vectors=vectors,
        steering_layers=settings.steering_layers,
        coefficients=settings.coefficients,
        prompts=prompts,
        seed=seed,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=settings.max_new_tokens,
    )
    json_path.write_text(
        json.dumps(generation_records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summarize_results(json_path, mode)


def move_vectors_to_device(
    vectors: TripletSteeringVectors, device: torch.device, dtype: torch.dtype
) -> TripletSteeringVectors:
    return TripletSteeringVectors(
        lou_minus_ja={pooling: vector.to(device, dtype=dtype) for pooling, vector in vectors.lou_minus_ja.items()},
        en_minus_ja={pooling: vector.to(device, dtype=dtype) for pooling, vector in vectors.en_minus_ja.items()},
        lou_parallel_en={pooling: vector.to(device, dtype=dtype) for pooling, vector in vectors.lou_parallel_en.items()},
        lou_perp_en={pooling: vector.to(device, dtype=dtype) for pooling, vector in vectors.lou_perp_en.items()},
    )


def generate_steering_comparison(
    model: Any,
    tokenizer: Any,
    vectors: TripletSteeringVectors,
    steering_layers: list[int],
    coefficients: list[float],
    prompts: list[str],
    seed: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    baseline_rows = generate_baseline_once(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        seed=seed,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )
    records = tag_baseline_rows(baseline_rows)
    steering_coefficients = [coefficient for coefficient in coefficients if coefficient != 0.0]

    for pooling in POOLING_METHODS:
        for direction, vector in {
            "lou_minus_ja": vectors.lou_minus_ja[pooling],
            "en_minus_ja": vectors.en_minus_ja[pooling],
            "lou_parallel_en": vectors.lou_parallel_en[pooling],
            "lou_perp_en": vectors.lou_perp_en[pooling],
        }.items():
            records.extend(
                generate_sweep(
                    model=model,
                    tokenizer=tokenizer,
                    vector=vector,
                    pooling=pooling,
                    direction=direction,
                    layers=steering_layers,
                    coefficients=steering_coefficients,
                    prompts=prompts,
                    seed=seed,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                )
            )
    return records


def generate_baseline_once(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    seed: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    rows = []
    for prompt_idx, prompt in enumerate(prompts):
        answer = generate_once(
            model,
            tokenizer,
            prompt,
            seed=seed + prompt_idx,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
        row = {
            "prompt": prompt,
            "answer": answer,
            "seed": seed + prompt_idx,
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            **score_output(answer),
        }
        rows.append(row)
        print(
            f"[baseline prompt={prompt_idx}] "
            f"ja={row['japanese_rate']:.3f} kat={row['katakana_rate']:.3f} en={row['english_rate']:.3f} :: {answer[:120]}"
        )
    return rows


def tag_baseline_rows(baseline_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "pooling": "baseline",
            "direction": "baseline",
            "layer": -1,
            "coefficient": 0.0,
            **row,
        }
        for row in baseline_rows
    ]


def generate_sweep(
    model: Any,
    tokenizer: Any,
    vector: SteeringVector | None,
    pooling: str,
    direction: str,
    layers: list[int],
    coefficients: list[float],
    prompts: list[str],
    seed: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    records = []
    for layer in layers:
        layer_vector = (
            None
            if vector is None
            else SteeringVector({layer: vector.layer_activations[layer]}, layer_type=LAYER_TYPE)
        )
        for coefficient in coefficients:
            context = (
                nullcontext()
                if layer_vector is None or coefficient == 0.0
                else layer_vector.apply(model, multiplier=coefficient, min_token_index=0)
            )
            with context:
                records.extend(
                    generate_prompt_batch(
                        model=model,
                        tokenizer=tokenizer,
                        prompts=prompts,
                        pooling=pooling,
                        direction=direction,
                        layer=layer,
                        coefficient=coefficient,
                        seed=seed,
                        temperature=temperature,
                        top_p=top_p,
                        max_new_tokens=max_new_tokens,
                    )
                )
    return records


def generate_prompt_batch(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    pooling: str,
    direction: str,
    layer: int,
    coefficient: float,
    seed: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    rows = []
    for prompt_idx, prompt in enumerate(prompts):
        answer = generate_once(
            model,
            tokenizer,
            prompt,
            seed=seed + prompt_idx,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
        row = {
            "pooling": pooling,
            "direction": direction,
            "prompt": prompt,
            "answer": answer,
            "layer": layer,
            "coefficient": coefficient,
            "seed": seed + prompt_idx,
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            **score_output(answer),
        }
        rows.append(row)
        print(
            f"[pooling={pooling} direction={direction} layer={layer} coef={coefficient} prompt={prompt_idx}] "
            f"ja={row['japanese_rate']:.3f} kat={row['katakana_rate']:.3f} en={row['english_rate']:.3f} :: {answer[:120]}"
        )
    return rows
