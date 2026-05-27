"""Live and headless trainer for the attend demo."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from core.checkpoints import RunPaths
from core.plotting import save_loss_curve
from core.torch_utils import torch_device
from core.utils import set_seed
from demos.attend.config import AttendConfig
from demos.attend.data import (
    DISPLAY_EXAMPLES,
    LABELS,
    MAX_SENTENCE_LENGTH,
    VOCAB,
    AgreementBatch,
    AgreementExample,
    batch_from_examples,
    make_batch,
)
from demos.attend.model import AgreementSelfAttentionModel


class AttendTrainer:
    def __init__(self, config: AttendConfig) -> None:
        self.config = config
        self.device = torch_device(config.device)
        self.reset(int(config.seed))

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.config.seed = int(seed)
        set_seed(int(self.config.seed))
        self.rng = np.random.default_rng(int(self.config.seed))
        self.model = AgreementSelfAttentionModel(
            vocab_size=len(VOCAB),
            sequence_length=MAX_SENTENCE_LENGTH,
            embedding_dim=int(self.config.embedding_dim),
            attention_dim=int(self.config.attention_dim),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.config.lr))
        self.loss_fn = nn.CrossEntropyLoss()
        self.losses: list[float] = []
        self.accuracies: list[float] = []
        self.last_loss: float | None = None
        self.step_count = 0
        self.preview_index = 0
        self.preview = DISPLAY_EXAMPLES[self.preview_index]
        self.highlighted_row = int(self.preview.mask_index)

    def _batch_to_torch(self, batch: AgreementBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(batch.token_ids).to(self.device),
            torch.from_numpy(batch.targets).to(self.device),
            torch.from_numpy(batch.mask_indices).to(self.device),
            torch.from_numpy(batch.token_mask).to(self.device),
        )

    def step(self, n_steps: int = 1) -> None:
        for _idx in range(max(1, int(n_steps))):
            batch = make_batch(
                self.rng,
                batch_size=max(1, int(self.config.batch_size)),
                trap_rate=float(self.config.trap_rate),
                force_traps=True,
            )
            token_ids, targets, mask_indices, token_mask = self._batch_to_torch(batch)
            logits, _attention = self.model(token_ids, mask_indices, token_mask)
            loss = self.loss_fn(logits, targets)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            self.last_loss = float(loss.detach().cpu())
            with torch.no_grad():
                acc = float((logits.argmax(dim=1) == targets).float().mean().item())
            self.losses.append(self.last_loss)
            self.accuracies.append(acc)
            self.step_count += 1
            interval = int(self.config.display_refresh_every)
            if interval > 0 and self.step_count % interval == 0:
                self.new_preview()

    def new_preview(self) -> None:
        self.preview_index = (int(self.preview_index) + 1) % len(DISPLAY_EXAMPLES)
        self.preview = DISPLAY_EXAMPLES[self.preview_index]
        self.highlighted_row = int(self.preview.mask_index)

    def cycle_attention_row(self, delta: int = 1) -> None:
        self.highlighted_row = (int(self.highlighted_row) + int(delta)) % len(self.preview.tokens)

    def _eval_accuracy(self) -> float:
        batch = make_batch(self.rng, batch_size=128, trap_rate=float(self.config.trap_rate), force_traps=True)
        token_ids, targets, mask_indices, token_mask = self._batch_to_torch(batch)
        with torch.no_grad():
            logits, _attention = self.model(token_ids, mask_indices, token_mask)
            return float((logits.argmax(dim=1) == targets).float().mean().item())

    @property
    def metrics(self) -> dict[str, object]:
        acc = self.accuracies[-1] if self.accuracies else self._eval_accuracy()
        return {
            "demo": "attend",
            "dataset": self.config.dataset,
            "step": int(self.step_count),
            "loss": self.last_loss,
            "accuracy": float(acc),
            "tokens": MAX_SENTENCE_LENGTH,
        }

    def _preview_batch(self) -> AgreementBatch:
        return batch_from_examples((self.preview,))

    def snapshot(self) -> dict[str, object]:
        batch = self._preview_batch()
        token_ids, targets, mask_indices, token_mask = self._batch_to_torch(batch)
        with torch.no_grad():
            logits, attention = self.model(token_ids, mask_indices, token_mask)
            probs = torch.softmax(logits, dim=-1)
        probs_np = probs[0].detach().cpu().numpy().astype(np.float32)
        example: AgreementExample = self.preview
        length = len(example.tokens)
        attention_np = attention[0, :length, :length].detach().cpu().numpy().astype(np.float32)
        pred = int(np.argmax(probs_np))
        target = int(targets[0].detach().cpu())
        highlighted_row = int(self.highlighted_row) % len(example.tokens)
        return {
            "tokens": example.tokens,
            "sentence": " ".join(example.tokens),
            "target": target,
            "target_label": LABELS[target],
            "pred": pred,
            "predicted_label": LABELS[pred],
            "correct": bool(pred == target),
            "probs": probs_np,
            "labels": LABELS,
            "attention": attention_np,
            "highlighted_row": highlighted_row,
            "highlighted_token": example.tokens[highlighted_row],
            "mask_index": int(example.mask_index),
            "subject_index": int(example.subject_index),
            "distractor_index": int(example.distractor_index),
            "subject": example.subject,
            "distractor": example.distractor,
            "subject_number": example.subject_number,
            "distractor_number": example.distractor_number,
            "template_id": int(example.template_id),
            "trap": bool(example.trap),
            "loss": self.last_loss,
            "accuracy": float(self.accuracies[-1]) if self.accuracies else self._eval_accuracy(),
            "step": int(self.step_count),
            "losses": np.asarray(self.losses, dtype=np.float32),
            "accuracies": np.asarray(self.accuracies, dtype=np.float32),
            "metrics": self.metrics,
        }

    def save(self, run_paths: RunPaths) -> None:
        torch.save(
            {
                "model_state": {key: value.detach().cpu() for key, value in self.model.state_dict().items()},
                "config": self.config.__dict__.copy(),
                "vocab": VOCAB,
            },
            run_paths.checkpoint_path,
        )
        np.savez(
            run_paths.artifact_dir / "attention_training.npz",
            losses=np.asarray(self.losses, dtype=np.float32),
            accuracies=np.asarray(self.accuracies, dtype=np.float32),
        )
        if self.losses:
            save_loss_curve(self.losses, run_paths.artifact_dir / "loss.png", title="attend loss")
