"""Run ``lerobot-train`` with a faster action-chunk query for image (non-video) datasets.

In LeRobot 0.6.1 + ``datasets`` 4.x, ``DatasetReader._query_hf_dataset`` fetches the
``action`` chunk (50 frames for SmolVLA) with ``hf_dataset[key][indices]``, which still
formats whole rows: every camera PNG of all 50 frames is decoded and then dropped
(153 decodes per sample instead of 3). Selecting the column first returns the same
tensors ~15x faster (342 -> 23 ms per sample, measured on our 3-camera dataset), so
training is no longer limited by data loading on a CPU with few cores.

    python -m so101_vla.lerobot_train_fast <the usual lerobot-train arguments>
"""

from __future__ import annotations

import torch
from lerobot.datasets.dataset_reader import DatasetReader


def _query_hf_dataset(self: DatasetReader, query_indices: dict[str, list[int]]) -> dict:
    """Same result as LeRobot's version, but reads only the queried column."""
    result: dict = {}
    for key, q_idx in query_indices.items():
        if key in self._meta.video_keys:
            continue
        rel = q_idx if self._absolute_to_relative_idx is None else [self._absolute_to_relative_idx[i] for i in q_idx]
        result[key] = torch.stack(self.hf_dataset.select_columns([key])[rel][key])
    return result


def patch() -> None:
    DatasetReader._query_hf_dataset = _query_hf_dataset


# At import time, not under __main__: lerobot-train's DataLoader workers are started with
# "spawn", which re-imports this module (as __mp_main__) without running the __main__ block.
patch()

if __name__ == "__main__":
    from lerobot.scripts.lerobot_train import main

    main()
