"""Live and headless trainer for the optim demo."""

from __future__ import annotations

import numpy as np

from core.checkpoints import RunPaths
from core.plotting import save_loss_curve
from core.utils import set_seed
from demos.optim.config import OptimConfig
from demos.optim.data import LANDSCAPES, landscape_grid, loss_and_grad


OPTIMIZERS = ("sgd", "momentum", "adam")


class OptimTrainer:
    def __init__(self, config: OptimConfig) -> None:
        self.config = config
        self._grid_cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self.reset(int(config.seed))

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.config.seed = int(seed)
        set_seed(int(self.config.seed))
        self.rng = np.random.default_rng(int(self.config.seed))
        self.point = np.asarray([float(self.config.start_x), float(self.config.start_y)], dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)
        self.m = np.zeros(2, dtype=np.float32)
        self.v = np.zeros(2, dtype=np.float32)
        self.adam_t = 0
        self.losses: list[float] = []
        self.trail: list[np.ndarray] = [self.point.copy()]
        self.last_loss = loss_and_grad(self.config.landscape, self.point)[0]
        self.step_count = 0

    def new_start(self) -> None:
        self.config.start_x = float(self.rng.uniform(-1.65, 1.65))
        self.config.start_y = float(self.rng.uniform(-1.65, 1.65))
        self.reset(int(self.config.seed))

    def cycle_optimizer(self, delta: int = 1) -> None:
        idx = OPTIMIZERS.index(str(self.config.optimizer)) if str(self.config.optimizer) in OPTIMIZERS else 0
        self.config.optimizer = OPTIMIZERS[(idx + int(delta)) % len(OPTIMIZERS)]
        self.reset(int(self.config.seed))

    def cycle_landscape(self, delta: int = 1) -> None:
        name = str(self.config.landscape)
        idx = LANDSCAPES.index(name) if name in LANDSCAPES else 0
        self.config.landscape = LANDSCAPES[(idx + int(delta)) % len(LANDSCAPES)]
        self.reset(int(self.config.seed))

    def scale_lr(self, factor: float) -> None:
        self.config.lr = float(np.clip(float(self.config.lr) * float(factor), 0.001, 0.8))

    def step(self, n_steps: int = 1) -> None:
        for _ in range(max(1, int(n_steps))):
            loss, grad = loss_and_grad(self.config.landscape, self.point)
            name = str(self.config.optimizer)
            lr = float(self.config.lr)
            if name == "sgd":
                update = -lr * grad
            elif name == "momentum":
                self.velocity = float(self.config.momentum) * self.velocity - lr * grad
                update = self.velocity
            else:
                self.adam_t += 1
                beta1, beta2 = 0.9, 0.999
                self.m = beta1 * self.m + (1.0 - beta1) * grad
                self.v = beta2 * self.v + (1.0 - beta2) * (grad * grad)
                m_hat = self.m / (1.0 - beta1**self.adam_t)
                v_hat = self.v / (1.0 - beta2**self.adam_t)
                update = -lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            self.point = np.clip(self.point + update.astype(np.float32), -2.2, 2.2)
            self.last_loss = float(loss_and_grad(self.config.landscape, self.point)[0])
            self.losses.append(self.last_loss)
            self.trail.append(self.point.copy())
            keep = max(2, int(self.config.trail_length))
            if len(self.trail) > keep:
                self.trail = self.trail[-keep:]
            self.step_count += 1

    @property
    def metrics(self) -> dict[str, object]:
        return {
            "demo": "optim",
            "dataset": self.config.dataset,
            "step": int(self.step_count),
            "loss": self.last_loss,
            "optimizer": str(self.config.optimizer),
            "landscape": str(self.config.landscape),
        }

    def _grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = (str(self.config.landscape), int(self.config.landscape_resolution))
        if key not in self._grid_cache:
            self._grid_cache[key] = landscape_grid(*key)
        return self._grid_cache[key]

    def snapshot(self) -> dict[str, object]:
        xs, ys, values = self._grid()
        return {
            "xs": xs,
            "ys": ys,
            "values": values,
            "point": self.point.copy(),
            "trail": np.asarray(self.trail, dtype=np.float32),
            "losses": np.asarray(self.losses, dtype=np.float32),
            "metrics": self.metrics,
        }

    def save(self, run_paths: RunPaths) -> None:
        np.savez(
            run_paths.artifact_dir / "optim_path.npz",
            point=self.point,
            trail=np.asarray(self.trail, dtype=np.float32),
            losses=np.asarray(self.losses, dtype=np.float32),
            landscape=str(self.config.landscape),
            optimizer=str(self.config.optimizer),
        )
        if self.losses:
            save_loss_curve(self.losses, run_paths.artifact_dir / "loss.png", title="optim loss")
