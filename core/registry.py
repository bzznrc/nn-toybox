"""Tiny explicit demo registry."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Callable

from core.config import CommonConfig
from core.datasets import (
    DIGIT_DATASET_KEYS,
    IMAGE_DATASET_KEYS,
    TEXT_DATASET_KEYS,
    dataset_display_name,
    dataset_display_names,
    dataset_key,
)
from core.generated_2d import CLASSIFICATION_DATASET_KEYS, DIFFUSION_DATASET_KEYS
from demos.attend.config import AttendConfig, add_attend_args
from demos.conv.config import ConvConfig, add_conv_args
from demos.diffuse.config import DiffuseConfig, add_diffuse_args
from demos.encode.config import EncodeConfig, add_encode_args
from demos.embed.config import EmbedConfig, add_embed_args
from demos.grad.config import GradConfig, add_grad_args
from demos.optim.config import OptimConfig, add_optim_args
from demos.trace.config import TraceConfig, add_trace_args


@dataclass(frozen=True)
class DemoSpec:
    name: str
    config_cls: type[CommonConfig]
    trainer_path: str
    renderer_path: str
    default_dataset: str
    dataset_keys: tuple[str, ...]
    add_cli_args: Callable[[object], None]

    def trainer_cls(self) -> type:
        module_name, class_name = self.trainer_path.rsplit(".", 1)
        return getattr(import_module(module_name), class_name)

    def renderer_cls(self) -> type:
        module_name, class_name = self.renderer_path.rsplit(".", 1)
        return getattr(import_module(module_name), class_name)

    @property
    def dataset_names(self) -> tuple[str, ...]:
        return dataset_display_names(self.dataset_keys)


DEMO_SPECS: dict[str, DemoSpec] = {
    "grad": DemoSpec(
        name="grad",
        config_cls=GradConfig,
        trainer_path="demos.grad.trainer.GradTrainer",
        renderer_path="demos.grad.renderer.GradRenderer",
        default_dataset=dataset_display_name("moons"),
        dataset_keys=("distributions", *CLASSIFICATION_DATASET_KEYS),
        add_cli_args=add_grad_args,
    ),
    "embed": DemoSpec(
        name="embed",
        config_cls=EmbedConfig,
        trainer_path="demos.embed.trainer.EmbedTrainer",
        renderer_path="demos.embed.renderer.EmbedRenderer",
        default_dataset=dataset_display_name("tiny_semantics"),
        dataset_keys=TEXT_DATASET_KEYS,
        add_cli_args=add_embed_args,
    ),
    "encode": DemoSpec(
        name="encode",
        config_cls=EncodeConfig,
        trainer_path="demos.encode.trainer.EncodeTrainer",
        renderer_path="demos.encode.renderer.EncodeRenderer",
        default_dataset=dataset_display_name("icons"),
        dataset_keys=IMAGE_DATASET_KEYS,
        add_cli_args=add_encode_args,
    ),
    "diffuse": DemoSpec(
        name="diffuse",
        config_cls=DiffuseConfig,
        trainer_path="demos.diffuse.trainer.DiffuseTrainer",
        renderer_path="demos.diffuse.renderer.DiffuseRenderer",
        default_dataset=dataset_display_name("gaussian_mixtures"),
        dataset_keys=("distributions", *DIFFUSION_DATASET_KEYS),
        add_cli_args=add_diffuse_args,
    ),
    "trace": DemoSpec(
        name="trace",
        config_cls=TraceConfig,
        trainer_path="demos.trace.trainer.TraceTrainer",
        renderer_path="demos.trace.renderer.TraceRenderer",
        default_dataset=dataset_display_name("digits8_mini"),
        dataset_keys=DIGIT_DATASET_KEYS,
        add_cli_args=add_trace_args,
    ),
    "conv": DemoSpec(
        name="conv",
        config_cls=ConvConfig,
        trainer_path="demos.conv.trainer.ConvTrainer",
        renderer_path="demos.conv.renderer.ConvRenderer",
        default_dataset=dataset_display_name("digits8_mini"),
        dataset_keys=DIGIT_DATASET_KEYS,
        add_cli_args=add_conv_args,
    ),
    "attend": DemoSpec(
        name="attend",
        config_cls=AttendConfig,
        trainer_path="demos.attend.trainer.AttendTrainer",
        renderer_path="demos.attend.renderer.AttendRenderer",
        default_dataset=dataset_display_name("subject_verb_agreement"),
        dataset_keys=("subject_verb_agreement",),
        add_cli_args=add_attend_args,
    ),
    "optim": DemoSpec(
        name="optim",
        config_cls=OptimConfig,
        trainer_path="demos.optim.trainer.OptimTrainer",
        renderer_path="demos.optim.renderer.OptimRenderer",
        default_dataset=dataset_display_name("loss_landscapes"),
        dataset_keys=("loss_landscapes",),
        add_cli_args=add_optim_args,
    ),
}


DEMO_ORDER = tuple(DEMO_SPECS.keys())
DEMO_ALIASES: dict[str, str] = {"autoencode": "encode"}


def normalize_demo(value: str) -> str:
    key = str(value).strip().lower().replace("-", "_")
    key = DEMO_ALIASES.get(key, key)
    if key not in DEMO_SPECS:
        valid = ", ".join(DEMO_ORDER)
        aliases = ", ".join(sorted(DEMO_ALIASES))
        raise ValueError(f"Unknown demo '{value}'. Valid demos: {valid}. Valid aliases: {aliases}")
    return key


def get_demo_spec(name: str) -> DemoSpec:
    return DEMO_SPECS[normalize_demo(name)]


def validate_demo_dataset(spec: DemoSpec, dataset: object) -> str:
    """Return normalized dataset name or fail on unsupported demo/dataset pairs."""

    key = dataset_key(dataset)
    if key not in spec.dataset_keys:
        valid = ", ".join(spec.dataset_names)
        raise ValueError(
            f"Demo '{spec.name}' does not support dataset '{dataset}'. "
            f"Valid dataset{'s' if len(spec.dataset_keys) != 1 else ''}: {valid}"
        )
    return dataset_display_name(key)
