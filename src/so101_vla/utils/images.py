"""Minimal image output with no extra dependencies (PNG via zlib)."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np


def save_png(path: str | Path, rgb: np.ndarray) -> Path:
    """Write an (H, W, 3) uint8 RGB array as a PNG file."""
    img = np.ascontiguousarray(rgb, dtype=np.uint8)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) uint8, got {img.shape}")
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))  # filter type 0 per row

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)
    return path


def tile(images: list[np.ndarray], cols: int = 2) -> np.ndarray:
    """Arrange same-size images in a grid (for quick visual checks)."""
    h, w, c = images[0].shape
    rows = -(-len(images) // cols)
    out = np.zeros((rows * h, cols * w, c), np.uint8)
    for i, im in enumerate(images):
        r, k = divmod(i, cols)
        out[r * h:(r + 1) * h, k * w:(k + 1) * w] = im
    return out
