"""Live and headless trainer for the conv demo."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from core.checkpoints import RunPaths
from core.config import PROJECT_ROOT
from core.plotting import save_loss_curve
from core.torch_utils import torch_device
from core.utils import set_seed
from demos.conv.config import ConvConfig
from demos.conv.model import TinyConvNet
from nn_toybox.shared.digits8 import Digits8Browser, load_digits8_split


def _relative_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


class ConvTrainer:
    def __init__(self, config: ConvConfig) -> None:
        self.config = config
        self.device = torch_device(config.device)
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
        self.model = TinyConvNet(channels=int(self.config.channels)).to(self.device)
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
            "demo": "conv",
            "dataset": self.config.dataset,
            "step": int(self.step_count),
            "loss": self.last_loss,
            "accuracy": self._accuracy(self.x_inference, self.y_inference),
            "train_accuracy": self._accuracy(self.x_train, self.y_train),
            "checkpoint_loaded": bool(self.loaded_checkpoint),
        }

    def _browse_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if str(self.config.example_split).lower() == "train":
            return self.train_images, self.train_labels
        return self.inference_images, self.inference_labels

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

    def toggle_feature_layer(self) -> None:
        self.config.selected_feature_layer = 2 if int(self.config.selected_feature_layer) == 1 else 1

    def cycle_noise(self) -> None:
        levels = (0.0, 0.08, 0.16, 0.28)
        current = float(self.config.noise_amount)
        idx = min(range(len(levels)), key=lambda item: abs(levels[item] - current))
        self.config.noise_amount = levels[(idx + 1) % len(levels)]

    def _example_image(self) -> tuple[np.ndarray, int, int]:
        images, labels = self._browse_arrays()
        browser = self._browser()
        idx = int(browser.selected_index) % max(1, len(images))
        image = images[idx].copy()
        if int(self.config.shift_x) or int(self.config.shift_y):
            image[0] = np.roll(image[0], shift=(int(self.config.shift_y), int(self.config.shift_x)), axis=(0, 1))
        noise = max(0.0, float(self.config.noise_amount))
        if noise > 0.0:
            image = np.clip(image + self.rng.normal(0.0, noise, size=image.shape).astype(np.float32), 0.0, 1.0)
        return image.astype(np.float32), int(labels[idx]), idx

    def snapshot(self) -> dict[str, object]:
        image, label, idx = self._example_image()
        with torch.no_grad():
            features = self.model.forward_features(torch.from_numpy(image[None, ...]).to(self.device))
        conv1 = features["conv1"][0].detach().cpu().numpy().astype(np.float32)
        conv2 = features["conv2"][0].detach().cpu().numpy().astype(np.float32)
        probs = features["probs"][0].detach().cpu().numpy().astype(np.float32)
        pred = int(np.argmax(probs))
        filters = self.model.conv1.weight.detach().cpu().numpy().astype(np.float32)[:, 0]
        return {
            "index": idx,
            "image": image,
            "label": label,
            "pred": pred,
            "correct": bool(pred == label),
            "probs": probs,
            "conv1": conv1,
            "conv2": conv2,
            "filters": filters,
            "selected_digit": int(self._browser().selected_digit),
            "variation_index": int(self._browser().variation_index),
            "losses": np.asarray(self.losses, dtype=np.float32),
            "metrics": self.metrics,
        }

    def save(self, run_paths: RunPaths) -> None:
        torch.save(
            {
                "model_state": {key: value.detach().cpu() for key, value in self.model.state_dict().items()},
                "config": self.config.__dict__.copy(),
            },
            run_paths.checkpoint_path,
        )
        np.savez(
            run_paths.artifact_dir / "conv_digits.npz",
            train_images=self.train_images,
            train_labels=self.train_labels,
            inference_images=self.inference_images,
            inference_labels=self.inference_labels,
            losses=np.asarray(self.losses, dtype=np.float32),
        )
        if self.losses:
            save_loss_curve(self.losses, run_paths.artifact_dir / "loss.png", title="conv loss")
