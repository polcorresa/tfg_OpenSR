# OpenSR Local Runner

This repository is set up to run official ESA OpenSR models on local GeoTIFF cutouts.

It supports these upstream workflows:

- `sen2sr_rgbn_x4`: `SEN2SRLite NonReference_RGBN_x4` for `B04,B03,B02,B08` to 2.5 m.
- `sen2sr_full_x4`: `SEN2SRLite main` for `B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12` to 2.5 m.
- `opensr_model_rgbn_x4`: `opensr-model` / LDSR-S2 for `B04,B03,B02,B08` to 2.5 m.

## Current Input Status

The current file `rawData/inputBarcelona.tif` was configured with this band order:

- `B02, B03, B04, B8A, B11, B12`

That band set is not accepted by any official 2.5 m OpenSR workflow.

What is missing:

- For `sen2sr_rgbn_x4` or `opensr_model_rgbn_x4`: missing `B08`.
- For `sen2sr_full_x4`: missing `B05, B06, B07, B08`.

So the repository is prepared to run, but it will correctly refuse to generate a 2.5 m output from the current 6-band subset until the missing bands are provided.

## Layout

- `configs/`: runnable YAML configs.
- `rawData/`: input GeoTIFFs.
- `scripts/`: PowerShell helpers for setup and execution.
- `src/opensr_pipeline/`: Python code for validation, model loading, tiling, and GeoTIFF IO.

## Setup

Use the PowerShell helper from the repository root:

```powershell
./scripts/setup_env.ps1
```

Default behavior:

- creates `.venv`
- installs CUDA 12.1 PyTorch wheels
- installs SEN2SR dependencies
- installs optional `opensr-model` dependencies

CPU-only setup:

```powershell
./scripts/setup_env.ps1 -CpuOnly
```

Skip `opensr-model` and keep only SEN2SR support:

```powershell
./scripts/setup_env.ps1 -SkipOpenSRModel
```

## Inspect The Current Input

```powershell
./scripts/inspect_input.ps1 -Config configs/barcelona.current.yaml
```

This prints raster metadata and a compatibility report for all configured workflows.

## Run Inference

```powershell
./scripts/run_inference.ps1 -Config configs/barcelona.current.yaml
```

With the current config, this is expected to stop with a validation error because the required bands are missing.

## When You Add More Bands

If you add `B08` and only want RGB+NIR 2.5 m output, start from:

- `configs/examples/sen2sr_rgbn.example.yaml`

If you add the full 10-band Sentinel-2 stack, start from:

- `configs/examples/sen2sr_full.example.yaml`

## Notes

- The local runner uses its own rectangular-safe tiling instead of `sen2sr.predict_large`, so it works on non-square GeoTIFF cutouts.
- Output GeoTIFFs are written with the source CRS and transform scaled by the SR factor.
- Reflectance values are assumed to be stored as integer values scaled by `10000` unless the config says otherwise.
