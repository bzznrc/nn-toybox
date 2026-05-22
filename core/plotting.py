"""Static Matplotlib exports for README assets and run inspection."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


PALETTE = ("#66ccc1", "#f08070", "#4285f4", "#66bb6a", "#ab47bc", "#d6bc85")


def _prepare_path(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def save_loss_curve(losses: Sequence[float], path: str | Path, title: str) -> None:
    target = _prepare_path(path)
    fig, ax = plt.subplots(figsize=(5.5, 3.0), dpi=140)
    ax.plot(np.asarray(losses, dtype=np.float32), color="#66ccc1", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(target)
    plt.close(fig)


def save_grad_plot(model: torch.nn.Module, points: np.ndarray, labels: np.ndarray, path: str | Path) -> None:
    target = _prepare_path(path)
    model.eval()
    grid_x, grid_y = np.meshgrid(np.linspace(-1.6, 1.6, 180), np.linspace(-1.6, 1.6, 180))
    grid = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(np.float32)
    with torch.no_grad():
        logits = model(torch.from_numpy(grid))
        pred = logits.argmax(dim=1).cpu().numpy().reshape(grid_x.shape)

    fig, ax = plt.subplots(figsize=(4.4, 4.4), dpi=150)
    ax.contourf(grid_x, grid_y, pred, levels=np.arange(pred.max() + 2) - 0.5, colors=PALETTE, alpha=0.2)
    for label in sorted(set(labels.tolist())):
        mask = labels == int(label)
        ax.scatter(points[mask, 0], points[mask, 1], s=14, c=PALETTE[int(label) % len(PALETTE)], edgecolors="#1d2024", linewidths=0.25)
    ax.set_aspect("equal")
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_title("decision boundary")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(target)
    plt.close(fig)


def save_embedding_plot(tokens: Sequence[str], coords: np.ndarray, path: str | Path) -> None:
    target = _prepare_path(path)
    fig, ax = plt.subplots(figsize=(5.0, 4.2), dpi=150)
    ax.scatter(coords[:, 0], coords[:, 1], s=55, c="#66ccc1", edgecolors="#266e69", linewidths=1)
    for token, xy in zip(tokens, coords):
        ax.text(float(xy[0]) + 0.015, float(xy[1]) + 0.015, str(token), fontsize=8)
    ax.set_title("embedding projection")
    ax.grid(alpha=0.16)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(target)
    plt.close(fig)


def save_reconstruction_grid(images: np.ndarray, recons: np.ndarray, path: str | Path, count: int = 10) -> None:
    target = _prepare_path(path)
    n = min(int(count), int(images.shape[0]))
    fig, axes = plt.subplots(2, n, figsize=(n * 0.75, 1.8), dpi=160)
    for idx in range(n):
        axes[0, idx].imshow(images[idx, 0], cmap="gray", vmin=0, vmax=1)
        axes[1, idx].imshow(recons[idx, 0], cmap="gray", vmin=0, vmax=1)
        axes[0, idx].axis("off")
        axes[1, idx].axis("off")
    axes[0, 0].set_ylabel("in", fontsize=8)
    axes[1, 0].set_ylabel("out", fontsize=8)
    fig.tight_layout(pad=0.2)
    fig.savefig(target)
    plt.close(fig)


def save_diffusion_plot(clean: np.ndarray, noised: np.ndarray, generated: np.ndarray, path: str | Path) -> None:
    target = _prepare_path(path)
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0), dpi=150)
    for ax, title, points, color in (
        (axes[0], "clean", clean, "#66ccc1"),
        (axes[1], "noised", noised, "#e8eaed"),
        (axes[2], "generated", generated, "#f08070"),
    ):
        ax.scatter(points[:, 0], points[:, 1], s=5, c=color, alpha=0.85, linewidths=0)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-2.0, 2.0)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(target)
    plt.close(fig)

