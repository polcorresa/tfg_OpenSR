from __future__ import annotations

import argparse
from pathlib import Path

import torch

from opensr_pipeline.band_sets import reorder_indices, resolve_workflow
from opensr_pipeline.config import load_config
from opensr_pipeline.geoio import read_raster, write_raster
from opensr_pipeline.model_loaders import load_model
from opensr_pipeline.tiling import predict_large


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an ESA OpenSR workflow on a local GeoTIFF.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    args = parser.parse_args()

    config = load_config(config_path=(PROJECT_ROOT / args.config).resolve(), project_root=PROJECT_ROOT)
    raster = read_raster(config.input.path)

    if raster.array.shape[0] != len(config.input.band_order):
        raise ValueError(
            "Configured band order length does not match the GeoTIFF band count. "
            f"Configured={len(config.input.band_order)} GeoTIFF={raster.array.shape[0]}"
        )

    workflow = resolve_workflow(config.workflow, config.input.band_order)
    indices = reorder_indices(config.input.band_order, workflow.model_band_order)

    device = resolve_device(config.runtime.device)
    model = load_model(
        spec=workflow,
        models_dir=config.models_dir,
        device=device,
        sampling_steps=config.runtime.sampling_steps,
    )

    input_array = raster.array[indices].astype("float32") / config.input.reflectance_scale
    input_tensor = torch.from_numpy(input_array)
    input_tensor = torch.nan_to_num(input_tensor, nan=0.0, posinf=0.0, neginf=0.0)

    sr_tensor = predict_large(
        image=input_tensor,
        predict_tile=model.predict_tile,
        tile_size=config.runtime.tile_size,
        overlap=config.runtime.overlap,
        device=device,
        scale_factor=workflow.scale_factor,
    )

    write_raster(
        path=config.output.path,
        sr_array=sr_tensor.numpy(),
        reference_profile=raster.profile,
        reference_transform=raster.transform,
        scale_factor=workflow.scale_factor,
        reflectance_scale=config.input.reflectance_scale,
        band_descriptions=workflow.model_band_order,
    )

    print(f"Wrote super-resolved GeoTIFF to {config.output.path}")
    print(f"Workflow: {workflow.name}")
    print(f"Device: {device}")


def resolve_device(requested_device: str) -> str:
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if requested_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Config requested CUDA but torch.cuda.is_available() is false.")

    if requested_device not in {"cpu", "cuda"}:
        raise ValueError("Runtime device must be one of: auto, cpu, cuda.")

    return requested_device


if __name__ == "__main__":
    main()