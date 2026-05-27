"""Arcade renderer for live diffuse snapshots."""

from __future__ import annotations

import numpy as np

from core.arcade_style import (
    COLOR_AQUA,
    COLOR_BRICK_RED,
    COLOR_CORAL,
    COLOR_DEEP_TEAL,
    COLOR_FOG_GRAY,
    COLOR_SLATE_GRAY,
)
from core.arcade_view import clipped_rect, draw_points, fit_rect, split_columns
from demos.diffuse.config import DiffuseConfig


class DiffuseRenderer:
    def __init__(self, config: DiffuseConfig) -> None:
        self.config = config

    def draw(self, snapshot: dict[str, object], window: object) -> None:
        layout = window.layout(secondary=True)
        main_rect = layout.main
        secondary_rect = layout.secondary if layout.secondary is not None else layout.text
        noisy_rect, generated_rect = (fit_rect(rect, aspect=1.0) for rect in split_columns(main_rect, 2))
        clean = np.asarray(snapshot["clean"], dtype=np.float32)
        noisy = np.asarray(snapshot["noisy"], dtype=np.float32)
        generated = np.asarray(snapshot["generated"], dtype=np.float32)
        trajectory = np.asarray(snapshot["trajectory"], dtype=np.float32)
        metrics = dict(snapshot.get("metrics", {}))
        step = int(metrics.get("step", 0))
        sample_step = int(metrics.get("sample_step", 0))
        frame = (max(0, step - sample_step) // 5) % max(1, len(trajectory))
        with clipped_rect(main_rect):
            draw_points(
                noisy,
                None,
                noisy_rect,
                marker="small",
                alpha=120,
                limit=max(1, int(self.config.max_noised_points)),
                color_pair=(COLOR_AQUA, COLOR_DEEP_TEAL),
            )
            draw_points(
                trajectory[frame],
                None,
                generated_rect,
                marker="small",
                alpha=220,
                limit=max(1, int(self.config.max_generated_points)),
                color_pair=(COLOR_CORAL, COLOR_BRICK_RED),
            )
        draw_points(
            clean,
            None,
            secondary_rect,
            marker="small",
            alpha=210,
            limit=max(1, int(self.config.max_clean_points)),
            color_pair=(COLOR_FOG_GRAY, COLOR_SLATE_GRAY),
        )
        window.draw_info(
            snapshot,
            secondary=True,
            extra=(
                f"trajectory frame: {frame + 1} / {max(1, len(trajectory))}",
                f"generated samples: {len(generated)}",
                f"preview timesteps: {int(self.config.sample_timesteps)}",
                f"sample refresh: {int(self.config.sample_refresh_every)} steps",
            ),
        )
