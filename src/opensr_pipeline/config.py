from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from opensr_pipeline.band_sets import canonicalize_band_order


@dataclass(frozen=True)
class InputConfig:
    path: Path
    band_order: list[str]
    reflectance_scale: float


@dataclass(frozen=True)
class OutputConfig:
    path: Path


@dataclass(frozen=True)
class RuntimeConfig:
    device: str
    tile_size: int
    overlap: int
    sampling_steps: int


@dataclass(frozen=True)
class AppConfig:
    workflow: str
    models_dir: Path
    input: InputConfig
    output: OutputConfig
    runtime: RuntimeConfig


def load_config(config_path: Path, project_root: Path) -> AppConfig:
    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)

    if not isinstance(raw_config, dict):
        raise ValueError("Config file must contain a YAML mapping.")

    input_section = _require_mapping(raw_config, "input")
    output_section = _require_mapping(raw_config, "output")
    runtime_section = _require_mapping(raw_config, "runtime")

    input_path = _resolve_path(project_root, input_section["path"])
    output_path = _resolve_path(project_root, output_section["path"])
    models_dir = _resolve_path(project_root, raw_config.get("models_dir", "models"))

    input_config = InputConfig(
        path=input_path,
        band_order=canonicalize_band_order(list(input_section["band_order"])),
        reflectance_scale=float(input_section.get("reflectance_scale", 10000.0)),
    )
    output_config = OutputConfig(path=output_path)
    runtime_config = RuntimeConfig(
        device=str(runtime_section.get("device", "auto")),
        tile_size=int(runtime_section.get("tile_size", 128)),
        overlap=int(runtime_section.get("overlap", 32)),
        sampling_steps=int(runtime_section.get("sampling_steps", 100)),
    )

    return AppConfig(
        workflow=str(raw_config.get("workflow", "auto")),
        models_dir=models_dir,
        input=input_config,
        output=output_config,
        runtime=runtime_config,
    )


def _require_mapping(raw_config: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw_config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Config key '{key}' must be a mapping.")
    return value


def _resolve_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()
