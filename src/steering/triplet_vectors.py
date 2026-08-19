from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from steering_vectors import SteeringVector, record_activations

from constants import LAYER_TYPE
from data.chat import representation_text

PoolingMethod = Literal["mean", "final"]


@dataclass(frozen=True)
class TripletSteeringVectors:
    lou_minus_ja: dict[str, SteeringVector]
    en_minus_ja: dict[str, SteeringVector]
    lou_parallel_en: dict[str, SteeringVector]
    lou_perp_en: dict[str, SteeringVector]


def non_padding_mask(encoded: dict[str, torch.Tensor], tokenizer: Any) -> torch.Tensor:
    if "attention_mask" in encoded:
        return encoded["attention_mask"][0].bool()
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        return torch.ones_like(encoded["input_ids"][0], dtype=torch.bool)
    return encoded["input_ids"][0].ne(pad_token_id)


def content_token_mask_from_offsets(
    tokenizer: Any, text: str, sentence: str, encoded: dict[str, torch.Tensor]
) -> torch.Tensor:
    content_start = text.find(sentence)
    if content_start < 0:
        raise ValueError(f"Could not find sentence in chat template text: {sentence}")
    content_end = content_start + len(sentence)

    offsets = encoded.get("offset_mapping")
    if offsets is None:
        return content_token_mask_by_prefix(tokenizer, sentence, encoded)
    offsets = offsets[0]

    input_ids = encoded["input_ids"][0]
    special_ids = set(tokenizer.all_special_ids)
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for idx, (start, end) in enumerate(offsets.tolist()):
        token_id = int(input_ids[idx].item())
        overlaps_content = start < content_end and end > content_start
        if overlaps_content and token_id not in special_ids:
            mask[idx] = True
    if not torch.any(mask):
        raise ValueError(f"No user-content tokens found for: {sentence}")
    return mask


def content_token_mask_by_prefix(
    tokenizer: Any, sentence: str, encoded: dict[str, torch.Tensor]
) -> torch.Tensor:
    # Fallback for non-fast tokenizers without offset mapping. Qwen tokenizers should
    # normally use the offset path above.
    from data.chat import representation_text

    full_ids = encoded["input_ids"][0]
    empty_text = representation_text(tokenizer, "")
    prefix_len = len(tokenizer(empty_text, add_special_tokens=False)["input_ids"])
    sentence_len = len(tokenizer(sentence, add_special_tokens=False)["input_ids"])
    special_ids = set(tokenizer.all_special_ids)
    mask = torch.zeros_like(full_ids, dtype=torch.bool)
    for idx in range(prefix_len, min(prefix_len + sentence_len, full_ids.shape[0])):
        if int(full_ids[idx].item()) not in special_ids:
            mask[idx] = True
    if not torch.any(mask):
        raise ValueError(f"No user-content tokens found for: {sentence}")
    return mask


def assistant_start_index(encoded: dict[str, torch.Tensor], tokenizer: Any) -> int:
    mask = non_padding_mask(encoded, tokenizer)
    indices = torch.nonzero(mask, as_tuple=False).flatten()
    if indices.numel() == 0:
        raise ValueError("No non-padding tokens found for assistant-start representation")
    return int(indices[-1].item())


def pool_hidden_states(
    hidden_states: torch.Tensor, content_mask: torch.Tensor, final_index: int
) -> dict[PoolingMethod, torch.Tensor]:
    content_hidden = hidden_states[content_mask]
    if content_hidden.shape[0] == 0:
        raise ValueError("No user-content tokens found for mean pooling")
    return {
        "mean": content_hidden.mean(dim=0).detach().cpu().to(torch.float32),
        "final": hidden_states[final_index].detach().cpu().to(torch.float32),
    }


@torch.no_grad()
def extract_sentence_representations(
    model: Any,
    tokenizer: Any,
    sentence: str,
    layers: list[int],
    layer_config: dict[str, Any],
    layer_type: str = LAYER_TYPE,
) -> dict[PoolingMethod, dict[int, torch.Tensor]]:
    text = representation_text(tokenizer, sentence)
    try:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except NotImplementedError:
        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)

    content_mask = content_token_mask_from_offsets(tokenizer, text, sentence, encoded).to(model.device)
    final_index = assistant_start_index(encoded, tokenizer)
    encoded_for_model = {
        key: value.to(model.device)
        for key, value in encoded.items()
        if key != "offset_mapping"
    }

    with record_activations(
        model,
        layer_type=layer_type,
        layer_config=layer_config,
        clone_activations=True,
        layer_nums=layers,
    ) as records:
        model(**encoded_for_model)

    pooled: dict[PoolingMethod, dict[int, torch.Tensor]] = {"mean": {}, "final": {}}
    for layer in layers:
        layer_hidden = records[layer][-1][0]
        layer_pooled = pool_hidden_states(layer_hidden, content_mask, final_index)
        pooled["mean"][layer] = layer_pooled["mean"]
        pooled["final"][layer] = layer_pooled["final"]
    return pooled


@torch.no_grad()
def train_triplet_steering_vectors(
    model: Any,
    tokenizer: Any,
    triplets: list[dict[str, str]],
    layers: list[int],
    layer_config: dict[str, Any],
    layer_type: str = LAYER_TYPE,
) -> TripletSteeringVectors:
    sums = {
        "mean": {"lou_minus_ja": {layer: None for layer in layers}, "en_minus_ja": {layer: None for layer in layers}},
        "final": {"lou_minus_ja": {layer: None for layer in layers}, "en_minus_ja": {layer: None for layer in layers}},
    }

    for idx, triplet in enumerate(triplets, start=1):
        reps = {
            "ja": extract_sentence_representations(model, tokenizer, triplet["ja"], layers, layer_config, layer_type),
            "lou": extract_sentence_representations(model, tokenizer, triplet["lou"], layers, layer_config, layer_type),
            "en": extract_sentence_representations(model, tokenizer, triplet["en"], layers, layer_config, layer_type),
        }
        for pooling in ("mean", "final"):
            for layer in layers:
                lou_diff = reps["lou"][pooling][layer] - reps["ja"][pooling][layer]
                en_diff = reps["en"][pooling][layer] - reps["ja"][pooling][layer]
                lou_sum = sums[pooling]["lou_minus_ja"][layer]
                en_sum = sums[pooling]["en_minus_ja"][layer]
                sums[pooling]["lou_minus_ja"][layer] = lou_diff if lou_sum is None else lou_sum + lou_diff
                sums[pooling]["en_minus_ja"][layer] = en_diff if en_sum is None else en_sum + en_diff
        if idx == 1 or idx == len(triplets) or idx % 10 == 0:
            print(f"Triplet hidden-state extraction: {idx}/{len(triplets)} triplets")

    lou_vectors: dict[str, SteeringVector] = {}
    en_vectors: dict[str, SteeringVector] = {}
    parallel_vectors: dict[str, SteeringVector] = {}
    perp_vectors: dict[str, SteeringVector] = {}
    for pooling in ("mean", "final"):
        lou_activations = {
            layer: sums[pooling]["lou_minus_ja"][layer] / len(triplets)
            for layer in layers
        }
        en_activations = {
            layer: sums[pooling]["en_minus_ja"][layer] / len(triplets)
            for layer in layers
        }
        parallel_activations, perp_activations = project_lou_against_en(
            lou_activations, en_activations
        )
        lou_vectors[pooling] = SteeringVector(lou_activations, layer_type=layer_type)
        en_vectors[pooling] = SteeringVector(en_activations, layer_type=layer_type)
        parallel_vectors[pooling] = SteeringVector(parallel_activations, layer_type=layer_type)
        perp_vectors[pooling] = SteeringVector(perp_activations, layer_type=layer_type)
    return TripletSteeringVectors(
        lou_minus_ja=lou_vectors,
        en_minus_ja=en_vectors,
        lou_parallel_en=parallel_vectors,
        lou_perp_en=perp_vectors,
    )


def project_lou_against_en(
    lou_activations: dict[int, torch.Tensor],
    en_activations: dict[int, torch.Tensor],
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    parallel: dict[int, torch.Tensor] = {}
    perpendicular: dict[int, torch.Tensor] = {}
    for layer, lou_vector in lou_activations.items():
        en_vector = en_activations[layer]
        denom = torch.dot(en_vector, en_vector)
        if torch.isclose(denom, torch.tensor(0.0, dtype=denom.dtype)):
            projected = torch.zeros_like(lou_vector)
        else:
            projected = (torch.dot(lou_vector, en_vector) / denom) * en_vector
        parallel[layer] = projected.detach().cpu().to(torch.float32)
        perpendicular[layer] = (lou_vector - projected).detach().cpu().to(torch.float32)
    return parallel, perpendicular


def cosine_similarity_by_layer(
    lou_vector: SteeringVector, en_vector: SteeringVector
) -> list[dict[str, float | int]]:
    rows = []
    for layer in sorted(lou_vector.layer_activations):
        lou = lou_vector.layer_activations[layer].detach().cpu().to(torch.float32)
        en = en_vector.layer_activations[layer].detach().cpu().to(torch.float32)
        rows.append(
            {
                "layer": layer,
                "cosine_similarity": float(F.cosine_similarity(lou, en, dim=0).item()),
            }
        )
    return rows




def projection_check_by_layer(
    perp_vector: SteeringVector, en_vector: SteeringVector
) -> list[dict[str, float | int]]:
    rows = []
    for layer in sorted(perp_vector.layer_activations):
        perp = perp_vector.layer_activations[layer].detach().cpu().to(torch.float32)
        en = en_vector.layer_activations[layer].detach().cpu().to(torch.float32)
        rows.append(
            {
                "layer": layer,
                "cosine_perp_en": float(F.cosine_similarity(perp, en, dim=0).item()),
                "perp_norm": float(torch.linalg.vector_norm(perp).item()),
                "en_norm": float(torch.linalg.vector_norm(en).item()),
            }
        )
    return rows


def save_vector(vector: SteeringVector, path: Path) -> None:
    payload = {
        "layer_type": vector.layer_type,
        "layer_activations": {
            str(k): v.detach().cpu().to(torch.float32)
            for k, v in vector.layer_activations.items()
        },
    }
    torch.save(payload, path)


def save_triplet_vectors(vectors: TripletSteeringVectors, out_dir: Path, mode: str) -> None:
    for pooling, vector in vectors.lou_minus_ja.items():
        save_vector(vector, out_dir / f"lou_minus_ja_{pooling}_{mode}.pt")
    for pooling, vector in vectors.en_minus_ja.items():
        save_vector(vector, out_dir / f"en_minus_ja_{pooling}_{mode}.pt")
    for pooling, vector in vectors.lou_parallel_en.items():
        save_vector(vector, out_dir / f"lou_parallel_en_{pooling}_{mode}.pt")
    for pooling, vector in vectors.lou_perp_en.items():
        save_vector(vector, out_dir / f"lou_perp_en_{pooling}_{mode}.pt")
