from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine


@dataclass(frozen=True)
class RasterData:
    array: np.ndarray
    profile: dict[str, Any]
    transform: Affine
    descriptions: tuple[str | None, ...]


def read_raster(path: Path) -> RasterData:
    with rasterio.open(path) as dataset:
        array = dataset.read()
        profile = dataset.profile.copy()
        transform = dataset.transform
        descriptions = dataset.descriptions

    return RasterData(
        array=array,
        profile=profile,
        transform=transform,
        descriptions=descriptions,
    )


def write_raster(
    path: Path,
    sr_array: np.ndarray,
    reference_profile: dict[str, Any],
    reference_transform: Affine,
    scale_factor: int,
    reflectance_scale: float,
    band_descriptions: tuple[str, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    output_array = np.clip(sr_array * reflectance_scale, 0, 65535).round().astype(np.uint16)
    output_transform = reference_transform * Affine.scale(1 / scale_factor, 1 / scale_factor)

    profile = reference_profile.copy()
    profile.update(
        {
            "count": int(output_array.shape[0]),
            "height": int(output_array.shape[1]),
            "width": int(output_array.shape[2]),
            "dtype": "uint16",
            "transform": output_transform,
            "compress": "deflate",
        }
    )

    with rasterio.open(path, "w", **profile) as dataset:
        dataset.write(output_array)
        dataset.descriptions = tuple(band_descriptions)
