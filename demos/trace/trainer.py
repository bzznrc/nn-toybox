"""Live and headless trainer for the trace demo."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from core.checkpoints import RunPaths
from core.config import PROJECT_ROOT
from core.plotting import save_loss_curve
from core.utils import set_seed
from demos.trace.config import TraceConfig
from demos.trace.model import TinyTraceMLP
from nn_toybox.shared.digits8 import Digits8Browser, load_digits8_split


def _torch_device(name: str) -> torch.device:
    if str(name).lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _relative_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _top_edges(src: np.ndarray, weight: np.ndarray, top_k: int) -> np.ndarray:
    src_arr = np.asarray(src, dtype=np.float32).reshape(-1)
    weight_arr = np.asarray(weight, dtype=np.float32)
    contrib = weight_arr * src_arr[None, :]
    flat = np.abs(contrib).reshape(-1)
    if flat.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    k = max(1, min(int(top_k), flat.size))
    order = np.argpartition(-flat, k - 1)[:k]
    order = order[np.argsort(-flat[order])]
    dst = order // weight_arr.shape[1]
    src_idx = order % weight_arr.shape[1]
    signed = contrib[dst, src_idx]
    strength = np.abs(signed)
    return np.column_stack([src_idx, dst, signed, strength]).astype(np.float32)


class TraceTrainer:
    def __init__(self, config: TraceConfig) -> None:
        self.config = config
        self.device = _torch_device(config.device)
        self.reset(int(config.seed))

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.config.seed = int(seed)
        set_seed(int(self.config.seed))
        self.rng = np.random.default_rng(int(self.config.seed))
        train = load_digits8_split("train", channel_first=True)
        inference = load_digits8_split("inference", channel_first=True)
        self.train_images = train.images
        self.train_labels = train.labels
        self.inference_images = inference.images
        self.inference_labels = inference.labels
        self.model = TinyTraceMLP(hidden1=int(self.config.hidden1), hidden2=int(self.config.hidden2)).to(self.device)
        self.loaded_checkpoint = self._try_load_checkpoint()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.config.lr))
        self.loss_fn = nn.CrossEntropyLoss()
        self.x_train = torch.from_numpy(self.train_images).to(self.device)
        self.y_train = torch.from_numpy(self.train_labels).to(self.device)
        self.x_inference = torch.from_numpy(self.inference_images).to(self.device)
        self.y_inference = torch.from_numpy(self.inference_labels).to(self.device)
        self.losses: list[float] = []
        self.last_loss: float | None = None
        self.step_count = 0
        self.train_browser = Digits8Browser(self.train_labels)
        self.inference_browser = Digits8Browser(self.inference_labels)

    def _try_load_checkpoint(self) -> bool:
        path_text = str(getattr(self.config, "checkpoint_path", "") or "")
        if not path_text:
            return False
        path = _relative_path(path_text)
        if not path.exists():
            return False
        payload = torch.load(path, map_location=self.device)
        state = payload.get("model_state", payload) if isinstance(payload, dict) else payload
        self.model.load_state_dict(state)
        return True

    def step(self, n_steps: int = 1) -> None:
        batch_size = min(len(self.train_images), max(1, int(self.config.batch_size)))
        for _ in range(max(1, int(n_steps))):
            batch = self.rng.choice(len(self.train_images), size=batch_size, replace=False)
            xb = self.x_train[batch]
            yb = self.y_train[batch]
            loss = self.loss_fn(self.model(xb), yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            self.last_loss = float(loss.detach().cpu())
            self.losses.append(self.last_loss)
            self.step_count += 1

    def _accuracy(self, x_tensor: torch.Tensor, y_tensor: torch.Tensor) -> float:
        with torch.no_grad():
            pred = self.model(x_tensor).argmax(dim=1)
            return float((pred == y_tensor).float().mean().item())

    @property
    def metrics(self) -> dict[str, object]:
        return {
            "demo": "trace",
            "dataset": self.config.dataset,
            "step": int(self.step_count),
            "loss": self.last_loss,
            "accuracy": self._accuracy(self.x_inference, self.y_inference),
            "train_accuracy": self._accuracy(self.x_train, self.y_train),
            "checkpoint_loaded": bool(self.loaded_checkpoint),
        }

    def _browse_arrays(self) -> tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
        if str(self.config.example_split).lower() == "train":
            return self.train_images, self.train_labels, self.x_train, self.y_train
        return self.inference_images, self.inference_labels, self.x_inference, self.y_inference

    def _browser(self) -> Digits8Browser:
        if str(self.config.example_split).lower() == "train":
            return self.train_browser
        return self.inference_browser

    @property
    def selected_index(self) -> int:
        return int(self._browser().selected_index)

    def next_example(self, delta: int = 1) -> None:
        self.next_variation(delta)

    def next_variation(self, delta: int = 1) -> None:
        self._browser().cycle_variation(int(delta))

    def random_example(self) -> None:
        self._browser().random(self.rng)

    def cycle_digit(self, delta: int = 1) -> None:
        self._browser().cycle_digit(int(delta))

    def clear_digit_filter(self) -> None:
        return None

    def _selected_trace(self) -> dict[str, object]:
        images, labels, x_tensor, _y_tensor = self._browse_arrays()
        browser = self._browser()
        idx = int(browser.selected_index) % max(1, len(images))
        with torch.no_grad():
            acts = self.model.forward_with_activations(x_tensor[idx : idx + 1])
        arrays = {key: value[0].detach().cpu().numpy().astype(np.float32) for key, value in acts.items()}
        probs = arrays["probs"]
        weights = [
            self.model.fc1.weight.detach().cpu().numpy().astype(np.float32),
            self.model.fc2.weight.detach().cpu().numpy().astype(np.float32),
            self.model.fc3.weight.detach().cpu().numpy().astype(np.float32),
        ]
        edge_budget = max(1, int(self.config.top_k_edges))
        top_edges = [
            _top_edges(arrays["input"], weights[0], edge_budget),
            _top_edges(arrays["hidden1"], weights[1], edge_budget),
            _top_edges(arrays["hidden2"], weights[2], edge_budget),
        ]
        pred = int(np.argmax(probs))
        label = int(labels[idx])
        return {
            "index": idx,
            "image": images[idx],
            "label": label,
            "pred": pred,
            "correct": bool(pred == label),
            "input": arrays["input"],
            "hidden1": arrays["hidden1"],
            "hidden2": arrays["hidden2"],
            "logits": arrays["logits"],
            "probs": probs,
            "weights": weights,
            "top_edges": top_edges,
            "selected_digit": int(browser.selected_digit),
            "variation_index": int(browser.variation_index),
            "split": str(self.config.example_split),
        }

    def snapshot(self) -> dict[str, object]:
        snap = self._selected_trace()
        snap["losses"] = np.asarray(self.losses, dtype=np.float32)
        snap["metrics"] = self.metrics
        return snap

    def save(self, run_paths: RunPaths) -> None:
        torch.save(
            {
                "model_state": {key: value.detach().cpu() for key, value in self.model.state_dict().items()},
                "config": self.config.__dict__.copy(),
            },
            run_paths.checkpoint_path,
        )
        np.savez(
            run_paths.artifact_dir / "trace_digits.npz",
            train_images=self.train_images,
            train_labels=self.train_labels,
            inference_images=self.inference_images,
            inference_labels=self.inference_labels,
            losses=np.asarray(self.losses, dtype=np.float32),
        )
        if self.losses:
            save_loss_curve(self.losses, run_paths.artifact_dir / "loss.png", title="trace loss")
