from __future__ import annotations

from importlib.resources import files
import os
import unittest

import numpy as np

from core.config import PROJECT_ROOT
from core.datasets import DATASET_ALIASES, dataset_display_name, dataset_key
from core.generated_2d import CLASSIFICATION_DATASET_KEYS, DIFFUSION_DATASET_KEYS
from core.registry import DEMO_ALIASES, DEMO_ORDER, get_demo_spec, validate_demo_dataset
from core.digits8 import Digits8Browser, Digits8Dataset, load_digits8_split
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
from demos.diffuse.config import DiffuseConfig
from demos.diffuse.trainer import DiffuseTrainer
from demos.embed.config import EmbedConfig
from demos.embed.trainer import EmbedTrainer
from demos.encode.config import EncodeConfig
from demos.encode.trainer import EncodeTrainer
from demos.grad.config import GradConfig
from demos.grad.trainer import GradTrainer
from demos.optim.config import OptimConfig
from demos.optim.data import LANDSCAPES, landscape_grid, loss_and_grad
from demos.optim.trainer import OptimTrainer
from demos.trace.config import TraceConfig
from demos.trace.trainer import TraceTrainer
from scripts.run import main as run_main
from scripts.capture_demo import AutoCycleController, _has_simple_convergence


class ConsolidationSmokeTests(unittest.TestCase):
    def test_packaged_digits8_resource_loads_without_cwd(self) -> None:
        resource = files("assets").joinpath("digits8_mini_8x8.zip")
        self.assertTrue(resource.is_file())
        old_cwd = os.getcwd()
        try:
            os.chdir(PROJECT_ROOT.parent)
            train = load_digits8_split("train")
            inference = load_digits8_split("inference")
        finally:
            os.chdir(old_cwd)
        self.assertEqual(train.images.shape, (500, 1, 8, 8))
        self.assertEqual(inference.images.shape, (100, 1, 8, 8))

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

    def test_demo_registry_and_dataset_aliases(self) -> None:
        self.assertEqual(DEMO_ORDER, ("grad", "embed", "encode", "diffuse", "trace", "conv", "attend", "optim"))
        self.assertEqual(DEMO_ALIASES["autoencode"], "encode")
        self.assertEqual(get_demo_spec("autoencode").name, "encode")
        alias_cases = {
            "moons": "Distributions - Moons",
            "circles": "Distributions - Circles",
            "spiral": "Distributions - Spiral",
            "spirals": "Distributions - Spiral",
            "rings": "Distributions - Rings",
            "gaussian-mixtures": "Distributions - Gaussian Mixtures",
            "gaussian_mixtures": "Distributions - Gaussian Mixtures",
            "mixtures": "Distributions - Gaussian Mixtures",
            "checkerboard": "Distributions - Checkerboard",
            "icons": "Images - Icons",
            "digits8": "Digits - 8x8 Mini",
            "digits": "Digits - 8x8 Mini",
            "8x8-digits": "Digits - 8x8 Mini",
            "tiny-semantics": "Text - Tiny Semantics",
            "tiny_semantics": "Text - Tiny Semantics",
            "big-tiny-semantics": "Text - Big Tiny Semantics",
            "big_tiny_semantics": "Text - Big Tiny Semantics",
        }
        self.assertEqual(set(alias_cases), set(DATASET_ALIASES))
        for alias, canonical in alias_cases.items():
            self.assertEqual(dataset_display_name(dataset_key(alias)), canonical)

        optim_spec = get_demo_spec("optim")
        with self.assertRaisesRegex(ValueError, "does not support dataset"):
            validate_demo_dataset(optim_spec, "Digits - 8x8 Mini")
        with self.assertRaisesRegex(ValueError, "Unknown dataset"):
            dataset_key("8x8 digits")
        self.assertEqual(dataset_key("Distributions - Moons"), "moons")
        self.assertEqual(validate_demo_dataset(optim_spec, optim_spec.default_dataset), "Optimization - Loss Landscapes")
        grad_spec = get_demo_spec("grad")
        self.assertEqual(validate_demo_dataset(grad_spec, "Distributions"), "Distributions")
        self.assertEqual(validate_demo_dataset(grad_spec, "moons"), "Distributions - Moons")
        trace_spec = get_demo_spec("trace")
        self.assertEqual(validate_demo_dataset(trace_spec, "digits8"), "Digits - 8x8 Mini")
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

    def test_grad_trainer_step_snapshot(self) -> None:
        config = GradConfig(
            demo="grad",
            dataset=dataset_display_name("moons"),
            n_points=64,
            batch_size=16,
            boundary_resolution=16,
            save_every=0,
        )
        trainer = GradTrainer(config)
        trainer.step(1)
        start_distribution = str(trainer.config.distribution)
        trainer.cycle_distribution(1)
        self.assertIn(str(trainer.config.distribution), CLASSIFICATION_DATASET_KEYS)
        self.assertNotEqual(str(trainer.config.distribution), start_distribution)
        snap = trainer.snapshot()
        self.assertEqual(snap["points"].shape[1], 2)
        self.assertEqual(snap["boundary_pred"].shape[0], 16 * 16)
        self.assertIn("accuracy", snap["metrics"])

    def test_embed_trainer_step_snapshot(self) -> None:
        config = EmbedConfig(
            demo="embed",
            dataset=dataset_display_name("tiny_semantics"),
            batch_size=16,
            embedding_dim=2,
            save_every=0,
        )
        trainer = EmbedTrainer(config)
        trainer.step(1)
        trainer.cycle_dataset(1)
        self.assertIn(str(trainer.config.dataset), get_demo_spec("embed").dataset_names)
        snap = trainer.snapshot()
        self.assertEqual(snap["coords"].shape[1], 2)
        self.assertGreater(len(snap["tokens"]), 0)
        self.assertIn("positive_pairs", snap["metrics"])

    def test_encode_trainer_step_snapshot(self) -> None:
        config = EncodeConfig(
            demo="encode",
            dataset=dataset_display_name("icons"),
            n_samples=32,
            batch_size=16,
            image_size=12,
            hidden_dim=16,
            latent_dim=4,
            save_every=0,
        )
        trainer = EncodeTrainer(config)
        trainer.step(1)
        trainer.cycle_latent_dim(1)
        snap = trainer.snapshot()
        self.assertEqual(snap["images"].shape[2:], (12, 12))
        self.assertEqual(snap["latents"].shape[1], int(trainer.config.latent_dim))
        self.assertIn("mean_error", snap["metrics"])

    def test_diffuse_trainer_step_snapshot(self) -> None:
        config = DiffuseConfig(
            demo="diffuse",
            dataset=dataset_display_name("gaussian_mixtures"),
            n_points=64,
            batch_size=16,
            hidden_dim=16,
            timesteps=4,
            sample_timesteps=4,
            sample_count=16,
            sample_refresh_every=1,
            max_clean_points=64,
            max_noised_points=64,
            max_generated_points=16,
            time_dim=8,
            save_every=0,
        )
        trainer = DiffuseTrainer(config)
        trainer.step(1)
        start_distribution = str(trainer.config.distribution)
        trainer.cycle_distribution(1)
        self.assertIn(str(trainer.config.distribution), DIFFUSION_DATASET_KEYS)
        self.assertNotEqual(str(trainer.config.distribution), start_distribution)
        snap = trainer.snapshot()
        self.assertEqual(snap["clean"].shape[1], 2)
        self.assertEqual(snap["generated"].shape[1], 2)
        self.assertIn("sample_timesteps", snap["metrics"])

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
        self.assertGreaterEqual(len(LANDSCAPES), 8)
        self.assertIn("hidden_well", LANDSCAPES)
        self.assertIn("narrow_pass", LANDSCAPES)
        self.assertIn("ripple_traps", LANDSCAPES)
        for landscape in LANDSCAPES:
            _loss, grad = loss_and_grad(landscape, np.asarray([0.35, -0.45], dtype=np.float32))
            self.assertTrue(bool(np.isfinite(grad).all()))
            _xs, _ys, values = landscape_grid(landscape, 16)
            self.assertEqual(values.shape, (16, 16))
            self.assertTrue(bool(np.isfinite(values).all()))
        config = OptimConfig(
            demo="optim",
            dataset=dataset_display_name("loss_landscapes"),
            steps=2,
            landscape_resolution=16,
            save_every=0,
        )
        trainer = OptimTrainer(config)
        first_landscape = str(trainer.config.landscape)
        trainer.cycle_landscape(1)
        self.assertNotEqual(str(trainer.config.landscape), first_landscape)
        trainer.step(2)
        snap = trainer.snapshot()
        self.assertEqual(snap["point"].shape, (2,))
        self.assertEqual(snap["trail"].shape[1], 2)
        self.assertEqual(snap["values"].shape, (16, 16))

    def test_shared_run_instantiates_every_demo(self) -> None:
        root = PROJECT_ROOT / "runs" / "_test_consolidation_smoke"
        root.mkdir(parents=True, exist_ok=True)
        runs = {
            "grad": ["--dataset", "moons", "--n-points", "64", "--batch-size", "16", "--boundary-resolution", "16"],
            "embed": ["--dataset", "tiny-semantics", "--batch-size", "16", "--embedding-dim", "2"],
            "encode": ["--dataset", "icons", "--n-samples", "32", "--batch-size", "16", "--image-size", "12"],
            "diffuse": [
                "--dataset",
                "gaussian-mixtures",
                "--n-points",
                "64",
                "--batch-size",
                "16",
                "--hidden-dim",
                "16",
                "--timesteps",
                "4",
                "--sample-timesteps",
                "4",
                "--sample-count",
                "16",
                "--sample-refresh-every",
                "1",
                "--time-dim",
                "8",
            ],
            "trace": ["--dataset", "digits8", "--batch-size", "16", "--top-k-edges", "8"],
            "conv": ["--dataset", "8x8-digits", "--batch-size", "16", "--channels", "4"],
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

    def test_shared_run_demo_alias(self) -> None:
        root = PROJECT_ROOT / "runs" / "_test_consolidation_smoke"
        run_main(
            [
                "--demo",
                "autoencode",
                "--dataset",
                "icons",
                "--steps",
                "1",
                "--save-every",
                "0",
                "--run-name",
                "autoencode_alias_unit",
                "--output-dir",
                str(root),
                "--n-samples",
                "32",
                "--batch-size",
                "16",
            ]
        )
        self.assertTrue((root / "encode" / "autoencode_alias_unit" / "metrics.json").exists())

    def test_capture_auto_cycle_convergence_gate(self) -> None:
        self.assertFalse(_has_simple_convergence("grad", {"accuracy": 0.50}, baseline_loss=1.0))
        self.assertTrue(_has_simple_convergence("grad", {"accuracy": 0.90}, baseline_loss=1.0))
        self.assertTrue(_has_simple_convergence("embed", {"loss": 0.80}, baseline_loss=1.0))
        self.assertTrue(_has_simple_convergence("encode", {"mean_error": 0.10}, baseline_loss=None))

        controller = AutoCycleController(demo="grad", fps=10, min_seconds=1.0, cooldown_seconds=2.0, max_cycles=1)
        self.assertFalse(controller.should_cycle(5, {"loss": 1.0, "accuracy": 0.95}))
        self.assertTrue(controller.should_cycle(10, {"loss": 0.8, "accuracy": 0.95}))
        controller.record_cycle(10)
        self.assertFalse(controller.should_cycle(31, {"loss": 0.5, "accuracy": 0.95}))


if __name__ == "__main__":
    unittest.main()
