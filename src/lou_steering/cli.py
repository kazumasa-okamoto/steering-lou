from __future__ import annotations

import argparse

from lou_steering.experiments.lou_qa import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "sweep", "extended"], default="smoke")
    args = parser.parse_args()
    run(args.mode)
