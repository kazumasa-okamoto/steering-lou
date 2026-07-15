from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from lou_steering.constants import OUT_DIR

RATIO_METRICS = {
    "japanese_rate": "Japanese ratio",
    "katakana_rate": "Katakana ratio",
    "english_rate": "English ratio",
}


def summarize_results(json_path: Path, mode: str, out_dir: Path = OUT_DIR) -> None:
    rows: list[dict[str, Any]] = json.loads(json_path.read_text(encoding="utf-8"))

    grouped: dict[tuple[str, str, int, float], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (row["pooling"], row["direction"], row["layer"], row["coefficient"]), []
        ).append(row)

    summary = []
    for (pooling, direction, layer, coefficient), group in sorted(grouped.items()):
        summary.append(
            {
                "pooling": pooling,
                "direction": direction,
                "layer": layer,
                "coefficient": coefficient,
                "mean_japanese_rate": mean(group, "japanese_rate"),
                "mean_katakana_rate": mean(group, "katakana_rate"),
                "mean_english_rate": mean(group, "english_rate"),
                "mean_metric_denominator": mean(group, "metric_denominator"),
            }
        )

    summary_path = out_dir / f"summary_{mode}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_ratio_sweeps(summary, mode, out_dir)


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(row[key] for row in rows) / len(rows)


def plot_ratio_sweeps(summary: list[dict[str, Any]], mode: str, out_dir: Path) -> None:
    baseline_rows = [row for row in summary if row["direction"] == "baseline"]
    steering_rows = [row for row in summary if row["direction"] != "baseline"]

    for pooling in sorted({row["pooling"] for row in steering_rows}):
        pooling_rows = [row for row in steering_rows if row["pooling"] == pooling]
        for layer in sorted({row["layer"] for row in pooling_rows}):
            layer_rows = [row for row in pooling_rows if row["layer"] == layer]
            coefficient_values = sorted({row["coefficient"] for row in layer_rows})
            for metric_key, ylabel in RATIO_METRICS.items():
                summary_key = f"mean_{metric_key}"
                plt.figure(figsize=(9, 5))
                if baseline_rows and coefficient_values:
                    baseline_y = mean(baseline_rows, summary_key)
                    plt.plot(
                        [min(coefficient_values), max(coefficient_values)],
                        [baseline_y, baseline_y],
                        linestyle="--",
                        color="black",
                        label="baseline",
                    )
                for direction in [
                    "lou_minus_ja",
                    "en_minus_ja",
                    "lou_parallel_en",
                    "lou_perp_en",
                ]:
                    direction_rows = sorted(
                        [row for row in layer_rows if row["direction"] == direction],
                        key=lambda row: row["coefficient"],
                    )
                    if not direction_rows:
                        continue
                    xs = [row["coefficient"] for row in direction_rows]
                    ys = [row[summary_key] for row in direction_rows]
                    plt.plot(xs, ys, marker="o", label=direction)
                plt.axvline(0, color="black", linewidth=0.8, alpha=0.5)
                plt.ylim(-0.03, 1.03)
                plt.xlabel("coefficient")
                plt.ylabel(ylabel)
                plt.title(f"{ylabel}: {pooling}, layer {layer}")
                plt.legend()
                plt.tight_layout()
                plt.savefig(
                    out_dir / f"ratio_{metric_key}_{pooling}_layer{layer}_{mode}.png",
                    dpi=160,
                )
                plt.close()


def save_cosine_summary(
    cosine_by_pooling: dict[str, list[dict[str, float | int]]],
    mode: str,
    out_dir: Path = OUT_DIR,
) -> None:
    rows = []
    for pooling, pooling_rows in cosine_by_pooling.items():
        for row in pooling_rows:
            rows.append({"pooling": pooling, **row})
    path = out_dir / f"cosine_lou_ja_vs_en_ja_{mode}.json"
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_cosine_similarity(rows, mode, out_dir)




def save_projection_summary(
    projection_by_pooling: dict[str, list[dict[str, float | int]]],
    mode: str,
    out_dir: Path = OUT_DIR,
) -> None:
    rows = []
    for pooling, pooling_rows in projection_by_pooling.items():
        for row in pooling_rows:
            rows.append({"pooling": pooling, **row})
    path = out_dir / f"projection_lou_perp_vs_en_ja_{mode}.json"
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_projection_check(rows, mode, out_dir)


def plot_projection_check(
    rows: list[dict[str, Any]], mode: str, out_dir: Path = OUT_DIR
) -> None:
    plt.figure(figsize=(10, 5))
    for pooling in sorted({row["pooling"] for row in rows}):
        pooling_rows = sorted([row for row in rows if row["pooling"] == pooling], key=lambda x: x["layer"])
        xs = [row["layer"] for row in pooling_rows]
        ys = [row["cosine_perp_en"] for row in pooling_rows]
        plt.plot(xs, ys, marker="o", label=pooling)
    plt.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    plt.xlabel("layer")
    plt.ylabel("cosine similarity")
    plt.title("orthogonality check: v_perp vs v_EN-JA")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"projection_lou_perp_vs_en_ja_{mode}.png", dpi=160)
    plt.close()


def plot_cosine_similarity(
    rows: list[dict[str, Any]], mode: str, out_dir: Path = OUT_DIR
) -> None:
    plt.figure(figsize=(10, 5))
    for pooling in sorted({row["pooling"] for row in rows}):
        pooling_rows = sorted([row for row in rows if row["pooling"] == pooling], key=lambda x: x["layer"])
        xs = [row["layer"] for row in pooling_rows]
        ys = [row["cosine_similarity"] for row in pooling_rows]
        plt.plot(xs, ys, marker="o", label=pooling)
    plt.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    plt.xlabel("layer")
    plt.ylabel("cosine similarity")
    plt.title("cosine similarity: v_Lou-JA vs v_EN-JA")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"cosine_lou_ja_vs_en_ja_{mode}.png", dpi=160)
    plt.close()
