from __future__ import annotations

import argparse
import json
from pathlib import Path

import rasterio

from opensr_pipeline.band_sets import compatibility_report
from opensr_pipeline.config import load_config
from opensr_pipeline.geoio import read_raster


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a local GeoTIFF and report OpenSR compatibility.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    args = parser.parse_args()

    config = load_config(config_path=(PROJECT_ROOT / args.config).resolve(), project_root=PROJECT_ROOT)
    raster = read_raster(config.input.path)

    with rasterio.open(config.input.path) as dataset:
        metadata = {
            "path": str(config.input.path),
            "band_count": dataset.count,
            "width": dataset.width,
            "height": dataset.height,
            "dtype": dataset.dtypes[0] if dataset.dtypes else None,
            "crs": str(dataset.crs),
            "resolution": dataset.res,
            "bounds": tuple(dataset.bounds),
            "band_descriptions": list(raster.descriptions),
            "configured_band_order": config.input.band_order,
            "workflow_compatibility": compatibility_report(config.input.band_order),
        }

    print(json.dumps(metadata, indent=2, default=str))


if __name__ == "__main__":
    main()
