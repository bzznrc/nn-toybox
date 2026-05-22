"""Arcade renderer for live diffuse snapshots."""

from __future__ import annotations

import numpy as np

from core.arcade_view import draw_panel_outline, draw_points
from demos.diffuse.config import DiffuseConfig


class DiffuseRenderer:
    def __init__(self, config: DiffuseConfig) -> None:
        self.config = config

    def draw(self, snapshot: dict[str, object], window: object) -> None:
        layout = window.layout(secondary=True)
        main_rect = layout.main
        secondary_rect = layout.secondary if layout.secondary is not None else layout.text
        draw_panel_outline(main_rect, "")
        draw_panel_outline(secondary_rect, "")
        clean = np.asarray(snapshot["clean"], dtype=np.float32)
        noisy = np.asarray(snapshot["noisy"], dtype=np.float32)
        generated = np.asarray(snapshot["generated"], dtype=np.float32)
        trajectory = np.asarray(snapshot["trajectory"], dtype=np.float32)
        metrics = dict(snapshot.get("metrics", {}))
        frame = (int(metrics.get("step", 0)) // 5) % max(1, len(trajectory))
        draw_points(trajectory[frame], None, main_rect, size=5.0, alpha=220, limit=900)
        draw_points(clean, None, secondary_rect, size=3.0, alpha=210, limit=900)
        draw_points(noisy, None, secondary_rect, size=3.0, alpha=80, limit=900)
        window.draw_info(
            snapshot,
            secondary=True,
            extra=(
                f"trajectory frame: {frame + 1} / {max(1, len(trajectory))}",
                f"generated samples: {len(generated)}",
                f"sample refresh: {int(self.config.sample_refresh_every)} steps",
            ),
        )
