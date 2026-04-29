# OpenSR Local Runner

This repository runs official ESA OpenSR models on local GeoTIFF cutouts.

Supported workflows:

| Workflow | Model | Bands | Output |
|---|---|---|---|
| `sen2sr_rgbn_x4` | SEN2SRLite NonReference_RGBN_x4 | B04, B03, B02, B08 | 2.5 m |
| `sen2sr_full_x4` | SEN2SRLite main | B02–B08, B8A, B11, B12 | 2.5 m |
| `opensr_model_rgbn_x4` | LDSR-S2 (opensr-model) | B04, B03, B02, B08 | 2.5 m |

## Current Input

`rawData/barcelona_terminal_s2.tif` — 12-band Sentinel-2 scene (B01–B09, B11, B12, reflectance ×10000).

This file is compatible with `sen2sr_full_x4` (requires B02–B08, B8A, B11, B12).

Active config: `configs/barcelona.current.yaml`

## Layout

```
configs/
  barcelona.current.yaml       # active run config
  examples/                    # template configs for other workflows
rawData/                       # input GeoTIFFs
scripts/                       # PowerShell helpers
src/opensr_pipeline/           # validation, model loading, tiling, GeoTIFF IO
```

## Setup

```powershell
./scripts/setup_env.ps1
```

Creates `.venv`, installs CUDA 12.1 PyTorch wheels, SEN2SR, and `opensr-model` dependencies.

| Flag | Effect |
|---|---|
| `-CpuOnly` | Install CPU-only PyTorch |
| `-SkipOpenSRModel` | Skip `opensr-model`, keep SEN2SR only |

## Inspect Input

```powershell
./scripts/inspect_input.ps1 -Config configs/barcelona.current.yaml
```

Prints raster metadata and a compatibility report for all workflows.

## Run Inference

```powershell
./scripts/run_inference.ps1 -Config configs/barcelona.current.yaml
```

Output is written to `outputs/barcelona_terminal_s2_sr.tif`.

## Writing Your Own Config

Copy an example config and adjust the paths and band order:

```yaml
workflow: sen2sr_full_x4      # or sen2sr_rgbn_x4, opensr_model_rgbn_x4, auto
models_dir: models

input:
  path: rawData/your_scene.tif
  band_order: [B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12]
  reflectance_scale: 10000.0

output:
  path: outputs/your_scene_sr.tif

runtime:
  device: auto          # auto | cpu | cuda
  tile_size: 128
  overlap: 32
  sampling_steps: 100   # diffusion steps (opensr_model_rgbn_x4 only)
```

Set `workflow: auto` to let the pipeline pick the best compatible workflow automatically.

## Notes

- The local runner uses its own rectangular-safe tiling instead of `sen2sr.predict_large`, so it works on non-square GeoTIFF cutouts.
- Output GeoTIFFs are written with the source CRS and transform scaled by the SR factor.
- Models are downloaded on first run from HuggingFace and cached in the `models/` directory.
- NaN and Inf values in the input are zeroed before inference.
