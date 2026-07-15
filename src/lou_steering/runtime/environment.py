from __future__ import annotations

import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from lou_steering.constants import OUT_DIR


def shell(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def save_environment(out_dir: Path = OUT_DIR) -> dict[str, Any]:
    env = {
        "nvidia_smi": shell(["nvidia-smi"]),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_bf16_supported": torch.cuda.is_bf16_supported()
        if torch.cuda.is_available()
        else None,
        "transformers": importlib.metadata.version("transformers"),
        "steering_vectors": importlib.metadata.version("steering-vectors"),
    }
    if torch.cuda.is_available():
        env["gpu_name"] = torch.cuda.get_device_name(0)
        env["gpu_capability"] = torch.cuda.get_device_capability(0)
    out_dir.mkdir(exist_ok=True)
    (out_dir / "environment.json").write_text(
        json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return env
