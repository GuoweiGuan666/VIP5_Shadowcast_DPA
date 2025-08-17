#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Command-line interface for running the DCIP-IEOS pipeline."""

import argparse
import json
import os

from .poison_pipeline import PoisonPipeline

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DCIP-IEOS attack")
    parser.add_argument("--input-pool", required=True, help="raw competition pool JSON")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "caches"),
        help="directory for cached artifacts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.input_pool, "r", encoding="utf-8") as f:
        pool = json.load(f)
    pipeline = PoisonPipeline(args.output_dir)
    pipeline.run(pool)


if __name__ == "__main__":
    main()