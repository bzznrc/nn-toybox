"""Run tiny non-visual smoke training for every active demo."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from core.registry import DEMO_ORDER
from scripts.train import main as run_main


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fast CPU-only training smoke checks")
    parser.add_argument("--demo", dest="demo", choices=DEMO_ORDER, default=None)
    parser.add_argument("--output-dir", "--runs-root", dest="output_dir", default=None)
    return parser.parse_args(argv)


def _args_for(demo: str, output_dir: str | None) -> list[str]:
    args = ["--demo", demo, "--steps", "2", "--run-name", f"{demo}_smoke", "--save-every", "0"]
    if demo == "grad":
        args.extend(["--dataset", "moons", "--n-points", "64", "--batch-size", "32"])
    if demo == "autoencode":
        args.extend(["--n-samples", "64", "--batch-size", "32"])
    if demo == "diffuse":
        args.extend(
            [
                "--dataset",
                "rings",
                "--n-points",
                "64",
                "--batch-size",
                "32",
                "--timesteps",
                "4",
                "--sample-count",
                "32",
                "--sample-refresh-every",
                "2",
            ]
        )
    if output_dir is not None:
        args.extend(["--output-dir", str(Path(output_dir))])
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    demos = (args.demo,) if args.demo else DEMO_ORDER
    for demo in demos:
        print(f">>> Smoke: {demo}")
        run_main(_args_for(demo, args.output_dir))


if __name__ == "__main__":
    main()
