"""Tiny explicit demo registry."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Callable

from core.config import CommonConfig
from demos.autoencode.config import AutoencodeConfig, add_autoencode_args
from demos.diffuse.config import DiffuseConfig, add_diffuse_args
from demos.embed.config import EmbedConfig, add_embed_args
from demos.grad.config import GradConfig, add_grad_args


@dataclass(frozen=True)
class DemoSpec:
    name: str
    config_cls: type[CommonConfig]
    trainer_path: str
    renderer_path: str
    default_dataset: str
    add_cli_args: Callable[[object], None]

    def trainer_cls(self) -> type:
        module_name, class_name = self.trainer_path.rsplit(".", 1)
        return getattr(import_module(module_name), class_name)

    def renderer_cls(self) -> type:
        module_name, class_name = self.renderer_path.rsplit(".", 1)
        return getattr(import_module(module_name), class_name)


DEMO_SPECS: dict[str, DemoSpec] = {
    "grad": DemoSpec(
        name="grad",
        config_cls=GradConfig,
        trainer_path="demos.grad.trainer.GradTrainer",
        renderer_path="demos.grad.renderer.GradRenderer",
        default_dataset="moons",
        add_cli_args=add_grad_args,
    ),
    "embed": DemoSpec(
        name="embed",
        config_cls=EmbedConfig,
        trainer_path="demos.embed.trainer.EmbedTrainer",
        renderer_path="demos.embed.renderer.EmbedRenderer",
        default_dataset="tiny_semantics",
        add_cli_args=add_embed_args,
    ),
    "autoencode": DemoSpec(
        name="autoencode",
        config_cls=AutoencodeConfig,
        trainer_path="demos.autoencode.trainer.AutoencodeTrainer",
        renderer_path="demos.autoencode.renderer.AutoencodeRenderer",
        default_dataset="icons",
        add_cli_args=add_autoencode_args,
    ),
    "diffuse": DemoSpec(
        name="diffuse",
        config_cls=DiffuseConfig,
        trainer_path="demos.diffuse.trainer.DiffuseTrainer",
        renderer_path="demos.diffuse.renderer.DiffuseRenderer",
        default_dataset="rings",
        add_cli_args=add_diffuse_args,
    ),
}


DEMO_ORDER = tuple(DEMO_SPECS.keys())


def normalize_demo(value: str) -> str:
    key = str(value).strip().lower().replace("-", "_")
    if key not in DEMO_SPECS:
        valid = ", ".join(DEMO_ORDER)
        raise ValueError(f"Unknown demo '{value}'. Valid: {valid}")
    return key


def get_demo_spec(name: str) -> DemoSpec:
    return DEMO_SPECS[normalize_demo(name)]
