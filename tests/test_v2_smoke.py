from __future__ import annotations

import unittest

import numpy as np

from core.config import PROJECT_ROOT
from core.datasets import dataset_display_name, dataset_key
from demos.attend.config import AttendConfig
from demos.attend.data import (
    DISPLAY_EXAMPLES,
    LABELS,
    MASK_TOKEN,
    MAX_SENTENCE_LENGTH,
    TEMPLATES,
    all_examples,
    has_trap_cases,
    make_batch,
)
from demos.attend.trainer import AttendTrainer
from demos.conv.config import ConvConfig
from demos.conv.trainer import ConvTrainer
from demos.optim.config import OptimConfig
from demos.optim.trainer import OptimTrainer
from demos.trace.config import TraceConfig
from demos.trace.trainer import TraceTrainer
from core.registry import DEMO_ORDER, get_demo_spec, validate_demo_dataset
from nn_toybox.shared.digits8 import Digits8Browser, Digits8Dataset, load_digits8_split
from scripts.train import main as run_main


class V2SmokeTests(unittest.TestCase):
    def test_digits8_zip_loader_counts(self) -> None:
        train = load_digits8_split("train")
        inference = load_digits8_split("inference")
        self.assertEqual(train.images.shape, (500, 1, 8, 8))
        self.assertEqual(inference.images.shape, (100, 1, 8, 8))
        self.assertEqual(train.labels.shape, (500,))
        self.assertEqual(inference.labels.shape, (100,))
        self.assertEqual(set(np.unique(train.labels).tolist()), set(range(10)))
        self.assertEqual(set(np.unique(inference.labels).tolist()), set(range(10)))
        dataset = Digits8Dataset("inference")
        image, label = dataset[0]
        self.assertEqual(image.shape, (1, 8, 8))
        self.assertIsInstance(label, int)

    def test_digits8_browser_arrow_semantics(self) -> None:
        labels = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
        browser = Digits8Browser(labels)
        self.assertEqual(browser.selected_index, 0)
        browser.cycle_variation(1)
        self.assertEqual(browser.selected_index, 1)
        browser.cycle_digit(1)
        self.assertEqual(browser.selected_digit, 1)
        self.assertEqual(browser.variation_index, 1)
        browser.cycle_digit(-1)
        self.assertEqual(browser.selected_digit, 0)
        self.assertEqual(browser.variation_index, 1)

    def test_demo_dataset_validation(self) -> None:
        optim_spec = get_demo_spec("optim")
        with self.assertRaisesRegex(ValueError, "does not support dataset"):
            validate_demo_dataset(optim_spec, "Digits - 8x8 Mini")
        with self.assertRaisesRegex(ValueError, "Unknown dataset"):
            dataset_key("8x8 digits")
        self.assertEqual(validate_demo_dataset(optim_spec, optim_spec.default_dataset), "Optimization - Loss Landscapes")
        trace_spec = get_demo_spec("trace")
        self.assertEqual(validate_demo_dataset(trace_spec, "Digits - 8x8 Mini"), "Digits - 8x8 Mini")
        for demo in DEMO_ORDER:
            spec = get_demo_spec(demo)
            self.assertEqual(validate_demo_dataset(spec, spec.default_dataset), spec.default_dataset)

    def test_attend_agreement_generator(self) -> None:
        examples = all_examples()
        self.assertTrue(has_trap_cases(examples))
        self.assertTrue(has_trap_cases(DISPLAY_EXAMPLES))
        self.assertEqual({example.template_id for example in examples}, set(range(1, len(TEMPLATES) + 1)))
        self.assertGreater(len({example.subject_index for example in examples}), 1)
        self.assertGreater(len({example.mask_index for example in examples}), 1)
        for example in DISPLAY_EXAMPLES:
            self.assertLessEqual(len(example.tokens), MAX_SENTENCE_LENGTH)
            self.assertEqual(example.tokens[example.mask_index], MASK_TOKEN)
            expected = "is" if example.subject_number == "singular" else "are"
            self.assertEqual(LABELS[example.target], expected)
            self.assertEqual(example.trap, example.subject_number != example.distractor_number)
        rng = np.random.default_rng(0)
        batch = make_batch(rng, batch_size=8)
        self.assertEqual(batch.token_ids.shape, (8, MAX_SENTENCE_LENGTH))
        self.assertEqual(batch.token_mask.shape, (8, MAX_SENTENCE_LENGTH))
        self.assertEqual(batch.targets.shape, (8,))
        self.assertIn(0, batch.targets.tolist())
        self.assertIn(1, batch.targets.tolist())
        self.assertTrue(bool(np.any(batch.traps)))

    def test_trace_trainer_step_snapshot(self) -> None:
        config = TraceConfig(
            demo="trace",
            dataset=dataset_display_name("digits8_mini"),
            batch_size=16,
            top_k_edges=12,
            save_every=0,
        )
        trainer = TraceTrainer(config)
        trainer.step(1)
        start_label = int(trainer.inference_labels[trainer.selected_index])
        trainer.next_variation(1)
        self.assertEqual(int(trainer.inference_labels[trainer.selected_index]), start_label)
        trainer.cycle_digit(1)
        self.assertNotEqual(int(trainer.inference_labels[trainer.selected_index]), start_label)
        snap = trainer.snapshot()
        self.assertEqual(snap["image"].shape, (1, 8, 8))
        self.assertEqual(snap["probs"].shape, (10,))
        self.assertEqual(len(snap["top_edges"]), 3)

    def test_conv_trainer_step_snapshot(self) -> None:
        config = ConvConfig(
            demo="conv",
            dataset=dataset_display_name("digits8_mini"),
            batch_size=16,
            channels=4,
            save_every=0,
        )
        trainer = ConvTrainer(config)
        trainer.step(1)
        start_label = int(trainer.inference_labels[trainer.selected_index])
        trainer.next_variation(1)
        self.assertEqual(int(trainer.inference_labels[trainer.selected_index]), start_label)
        trainer.cycle_digit(1)
        self.assertNotEqual(int(trainer.inference_labels[trainer.selected_index]), start_label)
        snap = trainer.snapshot()
        self.assertEqual(snap["image"].shape, (1, 8, 8))
        self.assertEqual(snap["conv1"].shape[1:], (8, 8))
        self.assertEqual(snap["probs"].shape, (10,))

    def test_attend_trainer_step_snapshot(self) -> None:
        config = AttendConfig(
            demo="attend",
            dataset=dataset_display_name("subject_verb_agreement"),
            batch_size=16,
            embedding_dim=8,
            attention_dim=8,
            save_every=0,
        )
        config = AttendConfig(**AttendConfig.resolve_payload(config.__dict__))
        trainer = AttendTrainer(config)
        trainer.step(1)
        snap = trainer.snapshot()
        self.assertLessEqual(len(snap["tokens"]), int(config.sequence_length))
        self.assertEqual(snap["probs"].shape, (2,))
        self.assertEqual(snap["attention"].shape, (len(snap["tokens"]), len(snap["tokens"])))
        self.assertEqual(snap["tokens"][int(snap["mask_index"])], MASK_TOKEN)
        self.assertIn("subject_index", snap)
        self.assertIn("distractor_index", snap)
        self.assertIn("template_id", snap)
        self.assertIn("trap", snap)
        self.assertIn(snap["target_label"], LABELS)
        self.assertIn(snap["predicted_label"], LABELS)
        self.assertGreaterEqual(float(np.max(snap["attention"])), 0.0)

    def test_optim_trainer_step_snapshot(self) -> None:
        config = OptimConfig(
            demo="optim",
            dataset=dataset_display_name("loss_landscapes"),
            steps=2,
            landscape_resolution=16,
            save_every=0,
        )
        trainer = OptimTrainer(config)
        trainer.step(2)
        snap = trainer.snapshot()
        self.assertEqual(snap["point"].shape, (2,))
        self.assertEqual(snap["trail"].shape[1], 2)
        self.assertEqual(snap["values"].shape, (16, 16))

    def test_shared_run_instantiates_new_demos(self) -> None:
        root = PROJECT_ROOT / "runs" / "_test_v2_smoke"
        root.mkdir(parents=True, exist_ok=True)
        runs = {
            "trace": ["--batch-size", "16", "--top-k-edges", "8"],
            "conv": ["--batch-size", "16", "--channels", "4"],
            "attend": ["--batch-size", "16", "--embedding-dim", "8", "--attention-dim", "8"],
            "optim": ["--landscape-resolution", "16", "--trail-length", "8"],
        }
        for demo, extra in runs.items():
            run_main(
                [
                    "--demo",
                    demo,
                    "--steps",
                    "1",
                    "--save-every",
                    "0",
                    "--run-name",
                    f"{demo}_unit",
                    "--output-dir",
                    str(root),
                    *extra,
                ]
            )
            self.assertTrue((root / demo / f"{demo}_unit" / "metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
