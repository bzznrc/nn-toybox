"""Run directory and checkpoint helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any

from core.config import CommonConfig, RUNS_ROOT


@dataclass(frozen=True)
class RunInfo:
    module: str
    dataset: str
    run_name: str
    run_dir: Path
    artifact_dir: Path
    checkpoint_path: Path
    metrics_path: Path
    config_path: Path


RunPaths = RunInfo


def _slug(value: str) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9_.-]+", "_", text)
    return text.strip("_") or "run"


def _runs_root(runs_root: str | Path | None = None) -> Path:
    return Path(runs_root).expanduser().resolve() if runs_root is not None else RUNS_ROOT


def runs_root(config: CommonConfig) -> Path:
    return Path(config.output_dir).expanduser().resolve() if config.output_dir else RUNS_ROOT


def create_run(
    module: str,
    dataset: str,
    *,
    run_name: str | None = None,
    runs_root: str | Path | None = None,
) -> RunInfo:
    root = _runs_root(runs_root)
    module_slug = _slug(module)
    dataset_slug = _slug(dataset)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_name = _slug(run_name) if run_name else f"{module_slug}_{dataset_slug}_{timestamp}"
    name = base_name
    run_dir = root / module_slug / name
    suffix = 2
    while run_dir.exists() and run_name is None:
        name = f"{base_name}_{suffix}"
        run_dir = root / module_slug / name
        suffix += 1
    artifact_dir = run_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (root / module_slug).mkdir(parents=True, exist_ok=True)
    (root / module_slug / "latest.txt").write_text(name, encoding="utf-8")
    return RunInfo(
        module=module_slug,
        dataset=dataset_slug,
        run_name=name,
        run_dir=run_dir,
        artifact_dir=artifact_dir,
        checkpoint_path=run_dir / "checkpoint.pt",
        metrics_path=run_dir / "metrics.json",
        config_path=run_dir / "config.json",
    )


def create_run_from_config(config: CommonConfig) -> RunInfo:
    return create_run(
        config.demo,
        config.dataset,
        run_name=config.run_name,
        runs_root=runs_root(config),
    )


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
