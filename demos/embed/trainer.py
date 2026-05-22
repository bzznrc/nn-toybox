"""Live and headless trainer for the embed demo."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from core.checkpoints import RunPaths
from core.plotting import save_embedding_plot, save_loss_curve
from core.utils import set_seed
from demos.embed.config import EmbedConfig
from demos.embed.data import relation_pairs
from demos.embed.model import TinyEmbeddingModel


def _project_2d(embeddings: np.ndarray) -> np.ndarray:
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    if centered.shape[1] == 1:
        return np.column_stack([centered[:, 0], np.zeros(centered.shape[0], dtype=np.float32)])
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T
    scale = np.max(np.linalg.norm(coords, axis=1))
    if scale > 0:
        coords = coords / scale
    return coords.astype(np.float32)


def _torch_device(name: str) -> torch.device:
    if str(name).lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class EmbedTrainer:
    def __init__(self, config: EmbedConfig) -> None:
        self.config = config
        self.device = _torch_device(config.device)
        self.reset(int(config.seed))

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.config.seed = int(seed)
        set_seed(int(self.config.seed))
        self.rng = np.random.default_rng(int(self.config.seed))
        self.tokens, positives, self.noisy_pairs = relation_pairs(
            self.config.dataset,
            noise_pairs=int(self.config.noise_pairs),
            seed=int(self.config.seed),
        )
        self.token_to_id = {token: idx for idx, token in enumerate(self.tokens)}
        self.pos_ids = np.asarray([(self.token_to_id[a], self.token_to_id[b]) for a, b in positives], dtype=np.int64)
        self.positive_set = {tuple(pair) for pair in self.pos_ids.tolist()}
        self.model = TinyEmbeddingModel(len(self.tokens), dim=int(self.config.embedding_dim)).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.config.lr))
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.losses: list[float] = []
        self.last_loss: float | None = None
        self.step_count = 0

    def _sample_pairs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        neg_count = max(1, len(self.pos_ids) * int(self.config.negative_samples))
        negatives: list[tuple[int, int]] = []
        while len(negatives) < neg_count:
            pair = (int(self.rng.integers(0, len(self.tokens))), int(self.rng.integers(0, len(self.tokens))))
            if pair[0] != pair[1] and pair not in self.positive_set:
                negatives.append(pair)
        neg_ids = np.asarray(negatives, dtype=np.int64)
        pairs = np.vstack([self.pos_ids, neg_ids])
        targets = np.concatenate([np.ones(len(self.pos_ids)), np.zeros(len(neg_ids))]).astype(np.float32)
        order = self.rng.permutation(len(pairs))
        left = torch.from_numpy(pairs[order, 0]).to(self.device)
        right = torch.from_numpy(pairs[order, 1]).to(self.device)
        y = torch.from_numpy(targets[order]).to(self.device)
        return left, right, y

    def step(self, n_steps: int = 1) -> None:
        for _ in range(max(1, int(n_steps))):
            left, right, targets = self._sample_pairs()
            loss = self.loss_fn(self.model(left, right), targets)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            self.last_loss = float(loss.detach().cpu())
            self.losses.append(self.last_loss)
            self.step_count += 1

    @property
    def metrics(self) -> dict[str, object]:
        return {
            "demo": "embed",
            "dataset": self.config.dataset,
            "step": int(self.step_count),
            "loss": self.last_loss,
            "tokens": len(self.tokens),
            "positive_pairs": len(self.pos_ids),
        }

    def _embeddings(self) -> np.ndarray:
        return self.model.embeddings().detach().cpu().numpy().astype(np.float32)

    def snapshot(self) -> dict[str, object]:
        embeddings = self._embeddings()
        return {
            "tokens": self.tokens,
            "embeddings": embeddings,
            "coords": _project_2d(embeddings),
            "positive_pairs": self.pos_ids,
            "losses": np.asarray(self.losses, dtype=np.float32),
            "metrics": self.metrics,
        }

    def save(self, run_paths: RunPaths) -> None:
        embeddings = self._embeddings()
        coords = _project_2d(embeddings)
        torch.save(
            {
                "model_state": {key: value.detach().cpu() for key, value in self.model.state_dict().items()},
                "config": self.config.__dict__.copy(),
                "tokens": self.tokens,
            },
            run_paths.checkpoint_path,
        )
        np.savez(
            run_paths.artifact_dir / "embeddings.npz",
            embeddings=embeddings,
            coords=coords,
            tokens=np.asarray(self.tokens),
            positives=np.asarray([(self.tokens[a], self.tokens[b]) for a, b in self.pos_ids.tolist()]),
            losses=np.asarray(self.losses, dtype=np.float32),
        )
        if self.losses:
            save_loss_curve(self.losses, run_paths.artifact_dir / "loss.png", title="embed loss")
        save_embedding_plot(self.tokens, coords, run_paths.artifact_dir / "projection.png")
