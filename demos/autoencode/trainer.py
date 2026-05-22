"""Live and headless trainer for the autoencode demo."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from core.checkpoints import RunPaths
from core.plotting import save_loss_curve, save_reconstruction_grid
from core.utils import set_seed
from demos.autoencode.config import AutoencodeConfig
from demos.autoencode.data import make_dataset
from demos.autoencode.model import TinyAutoencoder


def _torch_device(name: str) -> torch.device:
    if str(name).lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class AutoencodeTrainer:
    def __init__(self, config: AutoencodeConfig) -> None:
        self.config = config
        self.device = _torch_device(config.device)
        self.reset(int(config.seed))

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.config.seed = int(seed)
        set_seed(int(self.config.seed))
        self.rng = np.random.default_rng(int(self.config.seed))
        self.images, self.labels, self.names = make_dataset(
            self.config.dataset,
            n=int(self.config.n_samples),
            size=int(self.config.image_size),
            noise=float(self.config.noise),
            seed=int(self.config.seed),
        )
        input_dim = int(self.config.image_size) * int(self.config.image_size)
        self.model = TinyAutoencoder(
            input_dim=input_dim,
            latent_dim=int(self.config.latent_dim),
            hidden_size=int(self.config.hidden_dim),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.config.lr))
        self.loss_fn = nn.MSELoss()
        self.x_tensor = torch.from_numpy(self.images).to(self.device)
        self.losses: list[float] = []
        self.last_loss: float | None = None
        self.step_count = 0
        self.preview_indices = np.arange(min(8, len(self.images)))

    def step(self, n_steps: int = 1) -> None:
        batch_size = min(len(self.images), max(1, int(self.config.batch_size)))
        for _ in range(max(1, int(n_steps))):
            batch = self.rng.choice(len(self.images), size=batch_size, replace=False)
            xb = self.x_tensor[batch]
            recon, _z = self.model(xb)
            loss = self.loss_fn(recon, xb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            self.last_loss = float(loss.detach().cpu())
            self.losses.append(self.last_loss)
            self.step_count += 1

    def _preview(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        with torch.no_grad():
            sample = self.x_tensor[self.preview_indices]
            recons, latents = self.model(sample)
        return (
            sample.detach().cpu().numpy().astype(np.float32),
            recons.detach().cpu().numpy().astype(np.float32),
            latents.detach().cpu().numpy().astype(np.float32),
            self.labels[self.preview_indices],
        )

    @property
    def metrics(self) -> dict[str, object]:
        images, recons, _latents, _labels = self._preview()
        error = float(np.mean((images - recons) ** 2))
        return {
            "demo": "autoencode",
            "dataset": self.config.dataset,
            "step": int(self.step_count),
            "loss": self.last_loss,
            "mean_error": error,
            "latent_dim": int(self.config.latent_dim),
        }

    def snapshot(self) -> dict[str, object]:
        images, recons, latents, labels = self._preview()
        error = np.mean((images - recons) ** 2, axis=(1, 2, 3)).astype(np.float32)
        return {
            "images": images,
            "recons": recons,
            "latents": latents,
            "labels": labels,
            "names": tuple(self.names),
            "error": error,
            "losses": np.asarray(self.losses, dtype=np.float32),
            "metrics": self.metrics,
        }

    def _full_reconstruction(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        with torch.no_grad():
            recons, latents = self.model(self.x_tensor)
        recons_np = recons.detach().cpu().numpy().astype(np.float32)
        latents_np = latents.detach().cpu().numpy().astype(np.float32)
        error = np.mean((self.images - recons_np) ** 2, axis=(1, 2, 3)).astype(np.float32)
        return recons_np, latents_np, error, self.labels

    def save(self, run_paths: RunPaths) -> None:
        recons, latents, error, labels = self._full_reconstruction()
        torch.save(
            {
                "model_state": {key: value.detach().cpu() for key, value in self.model.state_dict().items()},
                "config": self.config.__dict__.copy(),
            },
            run_paths.checkpoint_path,
        )
        np.savez(
            run_paths.artifact_dir / "reconstructions.npz",
            images=self.images,
            recons=recons,
            latents=latents,
            labels=labels,
            names=np.asarray(self.names),
            losses=np.asarray(self.losses, dtype=np.float32),
            error=error,
        )
        if self.losses:
            save_loss_curve(self.losses, run_paths.artifact_dir / "loss.png", title="autoencode loss")
        save_reconstruction_grid(self.images, recons, run_paths.artifact_dir / "reconstructions.png")
