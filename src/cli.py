from __future__ import annotations

import argparse

from experiments.lou_qa import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "sweep"], default="smoke")
    args = parser.parse_args()
    run(args.mode)
