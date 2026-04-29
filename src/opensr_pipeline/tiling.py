from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F


def sliding_starts(size: int, tile_size: int, overlap: int) -> list[int]:
    if tile_size <= 0:
        raise ValueError("tile_size must be positive.")
    if overlap < 0:
        raise ValueError("overlap must be non-negative.")
    if overlap >= tile_size:
        raise ValueError("overlap must be smaller than tile_size.")

    if size <= tile_size:
        return [0]

    step = tile_size - overlap
    starts = list(range(0, size - tile_size + 1, step))
    if starts[-1] != size - tile_size:
        starts.append(size - tile_size)
    return starts


def iter_windows(height: int, width: int, tile_size: int, overlap: int) -> Iterable[tuple[int, int]]:
    for top in sliding_starts(height, tile_size, overlap):
        for left in sliding_starts(width, tile_size, overlap):
            yield top, left


def predict_large(
    image: torch.Tensor,
    predict_tile,
    tile_size: int,
    overlap: int,
    device: str,
    scale_factor: int,
) -> torch.Tensor:
    _, height, width = image.shape
    output: torch.Tensor | None = None
    weights = torch.zeros((1, height * scale_factor, width * scale_factor), dtype=torch.float32)

    for top, left in iter_windows(height, width, tile_size, overlap):
        tile = image[:, top : top + tile_size, left : left + tile_size]
        actual_height = tile.shape[1]
        actual_width = tile.shape[2]
        padded_tile = _pad_tile(tile, tile_size).to(device)

        prediction = predict_tile(padded_tile).detach().cpu()
        crop_height = actual_height * scale_factor
        crop_width = actual_width * scale_factor
        prediction = prediction[:, :crop_height, :crop_width]

        if output is None:
            output = torch.zeros(
                (prediction.shape[0], height * scale_factor, width * scale_factor),
                dtype=torch.float32,
            )

        output[
            :,
            top * scale_factor : top * scale_factor + crop_height,
            left * scale_factor : left * scale_factor + crop_width,
        ] += prediction
        weights[
            :,
            top * scale_factor : top * scale_factor + crop_height,
            left * scale_factor : left * scale_factor + crop_width,
        ] += 1.0

    if output is None:
        raise ValueError("No tiles were generated for prediction.")

    return output / weights.clamp_min(1.0)


def _pad_tile(tile: torch.Tensor, tile_size: int) -> torch.Tensor:
    pad_bottom = tile_size - tile.shape[1]
    pad_right = tile_size - tile.shape[2]
    if pad_bottom == 0 and pad_right == 0:
        return tile

    pad_mode = "reflect" if tile.shape[1] > 1 and tile.shape[2] > 1 else "replicate"
    padded = F.pad(tile[None], (0, pad_right, 0, pad_bottom), mode=pad_mode)
    return padded.squeeze(0)
