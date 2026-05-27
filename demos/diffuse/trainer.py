"""Live and headless trainer for the diffuse demo."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from core.checkpoints import RunPaths
from core.cycling import cycle_value
from core.generated_2d import DIFFUSION_DATASET_KEYS
from core.plotting import save_diffusion_plot, save_loss_curve
from core.torch_utils import torch_device
from core.utils import set_seed
from demos.diffuse.config import DiffuseConfig
from demos.diffuse.data import make_dataset
from demos.diffuse.model import TinyDenoiser, make_schedule, q_sample, sample_points


class DiffuseTrainer:
    def __init__(self, config: DiffuseConfig) -> None:
        self.config = config
        self.device = torch_device(config.device)
        self.reset(int(config.seed))

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.config.seed = int(seed)
        set_seed(int(self.config.seed))
        self.rng = np.random.default_rng(int(self.config.seed))
        self.clean = make_dataset(
            self.config.distribution,
            n=int(self.config.n_points),
            noise=float(self.config.noise),
            seed=int(self.config.seed),
        ).astype(np.float32)
        self.x_tensor = torch.from_numpy(self.clean).to(self.device)
        self.model = TinyDenoiser(hidden_size=int(self.config.hidden_dim), time_dim=int(self.config.time_dim)).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.config.lr))
        self.loss_fn = nn.MSELoss()
        _betas, _alphas, alphas_bar = make_schedule(int(self.config.timesteps), str(self.config.noise_schedule))
        self.alphas_bar = alphas_bar.to(self.device)
        self.losses: list[float] = []
        self.last_loss: float | None = None
        self.step_count = 0
        self._sample_step = -1
        self._generated = np.zeros((0, 2), dtype=np.float32)
        self._trajectory = np.zeros((1, 0, 2), dtype=np.float32)
        self._noisy = self._make_noisy()
        self._refresh_samples(force=True)

    def cycle_distribution(self, delta: int = 1) -> None:
        self.config.distribution = cycle_value(DIFFUSION_DATASET_KEYS, self.config.distribution, delta)
        self.reset(int(self.config.seed))

    def cycle_noise_schedule(self, delta: int = 1) -> None:
        self.config.noise_schedule = cycle_value(("linear", "cosine"), self.config.noise_schedule, delta)
        self.reset(int(self.config.seed))

    def _make_noisy(self) -> np.ndarray:
        with torch.inference_mode():
            t_last = torch.full((len(self.x_tensor),), int(self.config.timesteps) - 1, dtype=torch.long, device=self.device)
            noisy = q_sample(self.x_tensor, t_last, self.alphas_bar, torch.randn_like(self.x_tensor))
        return noisy.detach().cpu().numpy().astype(np.float32)

    def step(self, n_steps: int = 1) -> None:
        batch_size = min(len(self.clean), max(1, int(self.config.batch_size)))
        for _ in range(max(1, int(n_steps))):
            batch = self.rng.choice(len(self.clean), size=batch_size, replace=False)
            x0 = self.x_tensor[batch]
            t_idx_np = self.rng.integers(0, int(self.config.timesteps), size=len(batch)).astype(np.int64)
            t_idx = torch.from_numpy(t_idx_np).to(self.device)
            eps = torch.randn_like(x0)
            xt = q_sample(x0, t_idx, self.alphas_bar, eps)
            t_norm = t_idx.float() / max(1, int(self.config.timesteps) - 1)
            pred = self.model(xt, t_norm)
            loss = self.loss_fn(pred, eps)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            self.last_loss = float(loss.detach().cpu())
            self.losses.append(self.last_loss)
            self.step_count += 1

    def _refresh_samples(self, *, force: bool = False) -> None:
        cadence = max(1, int(self.config.sample_refresh_every))
        if not force and self._sample_step >= 0 and (self.step_count - self._sample_step) < cadence:
            return
        was_training = self.model.training
        self.model.eval()
        try:
            generated, trajectory = sample_points(
                self.model,
                count=max(1, min(int(self.config.sample_count), int(self.config.n_points))),
                steps=max(1, int(self.config.sample_timesteps)),
                schedule=str(self.config.noise_schedule),
                seed=int(self.config.seed) + int(self.step_count) + 1000,
                keep_frames=12,
                device=self.device,
            )
        finally:
            if was_training:
                self.model.train()
        self._generated = generated.detach().cpu().numpy().astype(np.float32)
        self._trajectory = trajectory.detach().cpu().numpy().astype(np.float32)
        self._sample_step = int(self.step_count)

    @property
    def metrics(self) -> dict[str, object]:
        return {
            "demo": "diffuse",
            "dataset": self.config.dataset,
            "distribution": self.config.distribution,
            "step": int(self.step_count),
            "sample_step": int(self._sample_step),
            "loss": self.last_loss,
            "timesteps": int(self.config.timesteps),
            "sample_timesteps": int(self.config.sample_timesteps),
            "noise_schedule": str(self.config.noise_schedule),
        }

    def snapshot(self) -> dict[str, object]:
        self._refresh_samples()
        return {
            "clean": self.clean,
            "noisy": self._noisy,
            "generated": self._generated,
            "trajectory": self._trajectory,
            "losses": np.asarray(self.losses, dtype=np.float32),
            "metrics": self.metrics,
        }

    def save(self, run_paths: RunPaths) -> None:
        self._refresh_samples(force=True)
        torch.save(
            {
                "model_state": {key: value.detach().cpu() for key, value in self.model.state_dict().items()},
                "config": self.config.__dict__.copy(),
            },
            run_paths.checkpoint_path,
        )
        np.savez(
            run_paths.artifact_dir / "samples.npz",
            clean=self.clean,
            noisy=self._noisy,
            generated=self._generated,
            trajectory=self._trajectory,
            losses=np.asarray(self.losses, dtype=np.float32),
        )
        if self.losses:
            save_loss_curve(self.losses, run_paths.artifact_dir / "loss.png", title="diffuse loss")
        save_diffusion_plot(self.clean, self._noisy[: len(self._generated)], self._generated, run_paths.artifact_dir / "samples.png")
