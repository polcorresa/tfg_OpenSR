from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Callable

import torch

from opensr_pipeline.band_sets import WorkflowSpec


OPEN_SR_MODEL_CONFIG_URL = "https://raw.githubusercontent.com/ESAOpenSR/opensr-model/refs/heads/main/opensr_model/configs/config_10m.yaml"


@dataclass(frozen=True)
class LoadedModel:
    predict_tile: Callable[[torch.Tensor], torch.Tensor]


def load_model(spec: WorkflowSpec, models_dir: Path, device: str, sampling_steps: int) -> LoadedModel:
    if spec.model_family == "sen2sr":
        return _load_sen2sr_model(spec=spec, models_dir=models_dir, device=device)

    if spec.model_family == "opensr_model":
        return _load_opensr_model(device=device, sampling_steps=sampling_steps)

    raise ValueError(f"Unsupported model family '{spec.model_family}'.")


def _load_sen2sr_model(spec: WorkflowSpec, models_dir: Path, device: str) -> LoadedModel:
    import mlstac

    if spec.model_url is None or spec.model_dir_name is None:
        raise ValueError(f"Workflow '{spec.name}' is missing SEN2SR model metadata.")

    models_dir.mkdir(parents=True, exist_ok=True)
    model_dir = models_dir / spec.model_dir_name
    if not model_dir.exists():
        mlstac.download(file=spec.model_url, output_dir=str(model_dir))

    compiled_model = mlstac.load(str(model_dir)).compiled_model(device=device)
    if hasattr(compiled_model, "to"):
        compiled_model = compiled_model.to(device)
    if hasattr(compiled_model, "eval"):
        compiled_model.eval()

    @torch.inference_mode()
    def predict_tile(tile: torch.Tensor) -> torch.Tensor:
        prediction = compiled_model(tile[None]).squeeze(0)
        return torch.nan_to_num(prediction, nan=0.0, posinf=0.0, neginf=0.0)

    return LoadedModel(predict_tile=predict_tile)


def _load_opensr_model(device: str, sampling_steps: int) -> LoadedModel:
    import requests
    from omegaconf import OmegaConf
    import opensr_model

    response = requests.get(OPEN_SR_MODEL_CONFIG_URL, timeout=30)
    response.raise_for_status()
    config = OmegaConf.load(StringIO(response.text))

    model = opensr_model.SRLatentDiffusion(config, device=device)
    model.load_pretrained(config.ckpt_version)
    if hasattr(model, "eval"):
        model.eval()

    @torch.inference_mode()
    def predict_tile(tile: torch.Tensor) -> torch.Tensor:
        prediction = model.forward(tile[None], sampling_steps=sampling_steps).squeeze(0)
        return torch.nan_to_num(prediction, nan=0.0, posinf=0.0, neginf=0.0)

    return LoadedModel(predict_tile=predict_tile)
