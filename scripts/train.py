"""Shared headless training/export entrypoint."""

from __future__ import annotations

import argparse
from typing import Sequence

from core.checkpoints import create_run_from_config, write_json
from core.config import add_common_args, config_from_args, to_dict
from core.registry import DEMO_ORDER, get_demo_spec, validate_demo_dataset


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    base = argparse.ArgumentParser(add_help=False)
    add_common_args(base, DEMO_ORDER, require_demo=False)
    known, _unknown = base.parse_known_args(argv)

    parser = argparse.ArgumentParser(description="Run an nn-toybox demo headlessly")
    add_common_args(parser, DEMO_ORDER)
    if known.demo is not None:
        spec = get_demo_spec(known.demo)
        spec.add_cli_args(parser)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    spec = get_demo_spec(args.demo)
    try:
        config = config_from_args(spec.config_cls, args, default_dataset=spec.default_dataset)
        config.dataset = validate_demo_dataset(spec, config.dataset)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    trainer = spec.trainer_cls()(config)
    run_paths = create_run_from_config(config)
    write_json(run_paths.config_path, to_dict(config))

    print(
        f"Run:\tdemo={config.demo}\tdataset={config.dataset}\tsteps={int(config.steps)}\trun={run_paths.run_dir}"
    )
    while trainer.step_count < int(config.steps):
        trainer.step(1)
        step = int(trainer.step_count)
        if step == 1 or step == int(config.steps) or step % int(config.log_every) == 0:
            metrics = trainer.metrics
            loss = metrics.get("loss")
            loss_text = "N/A" if loss is None else f"{float(loss):.5f}"
            extra = []
            for key in ("accuracy", "mean_error"):
                if key in metrics:
                    extra.append(f"{key}={float(metrics[key]):.3f}")
            suffix = "" if not extra else "\t" + "\t".join(extra)
            print(f"Step:\t{step:5d}\tLoss:\t{loss_text}{suffix}")
        if int(config.save_every) > 0 and trainer.step_count % int(config.save_every) == 0:
            trainer.save(run_paths)

    trainer.save(run_paths)
    write_json(run_paths.metrics_path, trainer.metrics)
    print(f">>> Run: {run_paths.run_dir}")


if __name__ == "__main__":
    main()
