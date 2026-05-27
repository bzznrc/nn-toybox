"""Live and headless trainer for the grad demo."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from core.checkpoints import RunPaths
from core.cycling import cycle_value
from core.plotting import save_grad_plot, save_loss_curve
from core.torch_utils import torch_device
from core.utils import set_seed
from demos.grad.config import GradConfig
from demos.grad.data import make_dataset
from demos.grad.model import TinyMLP
from core.generated_2d import CLASSIFICATION_DATASET_KEYS


class GradTrainer:
    def __init__(self, config: GradConfig) -> None:
        self.config = config
        self.device = torch_device(config.device)
        self.reset(int(config.seed))

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.config.seed = int(seed)
        set_seed(int(self.config.seed))
        self.rng = np.random.default_rng(int(self.config.seed))
        self.points, self.labels = make_dataset(
            self.config.distribution,
            n=int(self.config.n_points),
            noise=float(self.config.noise),
            seed=int(self.config.seed),
        )
        self.num_classes = int(self.labels.max()) + 1
        self.model = TinyMLP(
            hidden_size=int(self.config.hidden_dim),
            hidden_layers=int(self.config.layers),
            num_classes=self.num_classes,
            activation=str(self.config.activation),
        ).to(self.device)
        self.optimizer = self._make_optimizer()
        self.loss_fn = nn.CrossEntropyLoss()
        self.x_tensor = torch.from_numpy(self.points).to(self.device)
        self.y_tensor = torch.from_numpy(self.labels).to(self.device)
        self.losses: list[float] = []
        self.step_count = 0
        self.last_loss: float | None = None

    def _make_optimizer(self) -> torch.optim.Optimizer:
        if str(self.config.optimizer).lower() == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=float(self.config.lr), momentum=0.9)
        return torch.optim.Adam(self.model.parameters(), lr=float(self.config.lr))

    def cycle_distribution(self, delta: int = 1) -> None:
        self.config.distribution = cycle_value(CLASSIFICATION_DATASET_KEYS, self.config.distribution, delta)
        self.reset(int(self.config.seed))

    def cycle_optimizer(self, delta: int = 1) -> None:
        self.config.optimizer = cycle_value(("adam", "sgd"), self.config.optimizer, delta)
        self.reset(int(self.config.seed))

    def step(self, n_steps: int = 1) -> None:
        batch_size = min(len(self.points), max(1, int(self.config.batch_size)))
        for _ in range(max(1, int(n_steps))):
            batch = self.rng.choice(len(self.points), size=batch_size, replace=False)
            xb = self.x_tensor[batch]
            yb = self.y_tensor[batch]
            loss = self.loss_fn(self.model(xb), yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            self.last_loss = float(loss.detach().cpu())
            self.losses.append(self.last_loss)
            self.step_count += 1

    def _accuracy(self) -> float:
        with torch.no_grad():
            pred = self.model(self.x_tensor).argmax(dim=1)
            return float((pred == self.y_tensor).float().mean().item())

    @property
    def metrics(self) -> dict[str, object]:
        return {
            "demo": "grad",
            "dataset": self.config.dataset,
            "distribution": self.config.distribution,
            "step": int(self.step_count),
            "loss": self.last_loss,
            "accuracy": self._accuracy(),
        }

    def _boundary(self) -> tuple[np.ndarray, np.ndarray]:
        resolution = max(12, int(self.config.boundary_resolution))
        xs = np.linspace(-1.6, 1.6, resolution)
        ys = np.linspace(-1.6, 1.6, resolution)
        gx, gy = np.meshgrid(xs, ys)
        grid = np.column_stack([gx.ravel(), gy.ravel()]).astype(np.float32)
        with torch.no_grad():
            pred = self.model(torch.from_numpy(grid).to(self.device)).argmax(dim=1).cpu().numpy()
        return grid, pred

    def snapshot(self) -> dict[str, object]:
        grid, pred = self._boundary()
        return {
            "points": self.points,
            "labels": self.labels,
            "boundary_grid": grid,
            "boundary_pred": pred,
            "boundary_resolution": int(self.config.boundary_resolution),
            "losses": np.asarray(self.losses, dtype=np.float32),
            "metrics": self.metrics,
        }

    def _cpu_model(self) -> TinyMLP:
        model = TinyMLP(
            hidden_size=int(self.config.hidden_dim),
            hidden_layers=int(self.config.layers),
            num_classes=self.num_classes,
            activation=str(self.config.activation),
        )
        model.load_state_dict({key: value.detach().cpu() for key, value in self.model.state_dict().items()})
        model.eval()
        return model

    def save(self, run_paths: RunPaths) -> None:
        state = {key: value.detach().cpu() for key, value in self.model.state_dict().items()}
        torch.save(
            {
                "model_state": state,
                "config": self.config.__dict__.copy(),
                "num_classes": self.num_classes,
            },
            run_paths.checkpoint_path,
        )
        np.savez(
            run_paths.artifact_dir / "data.npz",
            points=self.points,
            labels=self.labels,
            losses=np.asarray(self.losses, dtype=np.float32),
        )
        if self.losses:
            save_loss_curve(self.losses, run_paths.artifact_dir / "loss.png", title="grad loss")
        save_grad_plot(self._cpu_model(), self.points, self.labels, run_paths.artifact_dir / "decision_boundary.png")
