from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RunConfig:
    mode: str
    model_id: str
    layers: list[int]
    steering_layers: list[int]
    coefficients: list[float]
    triplet_count: int
    prompt_count: int
    seed: int
    temperature: float
    top_p: float
    max_new_tokens: int
    layer_type: str
    residual_matcher: str
    pooling_methods: list[str]
    system_prompt: str


@dataclass(frozen=True)
class ModeSettings:
    triplet_limit: int | None
    steering_layers: list[int]
    coefficients: list[float]
    prompt_count: int
    max_new_tokens: int


def choose_layers(num_layers: int, positions: list[float] | None = None) -> list[int]:
    positions = positions or [0.30, 0.45, 0.60, 0.75]
    return sorted(
        {min(num_layers - 1, max(0, round((num_layers - 1) * p))) for p in positions}
    )


def mode_settings(mode: str, selected_layers: list[int]) -> ModeSettings:
    if mode == "smoke":
        return ModeSettings(
            triplet_limit=6,
            steering_layers=[selected_layers[len(selected_layers) // 2]],
            coefficients=[0.0, 0.1, 0.25, 0.5],
            prompt_count=1,
            max_new_tokens=160,
        )
    if mode != "sweep":
        raise ValueError(f"unsupported mode: {mode}")
    return ModeSettings(
        triplet_limit=None,
        steering_layers=selected_layers,
        coefficients=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
        prompt_count=50,
        max_new_tokens=160,
    )
