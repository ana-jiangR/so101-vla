"""Pretrained SmolVLA (``lerobot/smolvla_base``) through LeRobot's own APIs (Task 2.3).

Every step follows an official source; nothing here re-implements the model:

* loading: the model card (huggingface.co/lerobot/smolvla_base):
  ``SmolVLAPolicy.from_pretrained`` + ``make_pre_post_processors``
* per-step inference: ``lerobot-eval``'s ``rollout`` (lerobot/scripts/lerobot_eval.py):
  ``preprocess_observation`` -> ``observation["task"]`` -> preprocessor ->
  ``policy`` -> postprocessor, with the ``device_processor`` and
  ``rename_observations_processor`` overrides ``lerobot-eval`` passes
* camera order: the SmolVLA paper (arXiv 2506.01844, Sec. 3.2) standardized
  pretraining views as image 1 = top, 2 = wrist, 3 = side
* stats override: ``NormalizerProcessorStep`` (lerobot/processor/normalize_processor.py)
  keeps ``stats`` passed as a constructor override instead of the saved file

Verified finding (see docs/stage2.md): the checkpoint saves only *action* stats, under
the legacy keys ``so100*.buffer.action.{mean,std}``. LeRobot 0.6.1 loads them under
``so100.buffer.action`` etc., never under ``action`` / ``observation.state``, so the
as-shipped pipeline normalizes nothing. ``stats_mode`` makes this an explicit choice.

Import cost: torch/lerobot are imported lazily, so Stage 1 code never needs them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from so101_vla.conventions import LEROBOT_FEATURE_NAMES

STATE_KEY = "observation.state"
IMAGE_PREFIX = "observation.images."

# SmolVLA paper, Sec. 3.2: OBS_IMAGE_1/2/3 = top, wrist, side (named camera1..3 in config.json).
DEFAULT_CAMERA_MAP: dict[str, str] = {"camera1": "top", "camera2": "main/wrist_cam", "camera3": "front"}

STATS_MODES = ("as_shipped", "checkpoint_action_stats")


@dataclass(frozen=True)
class SmolVLAConfig:
    checkpoint: str = "lerobot/smolvla_base"
    revision: str | None = "5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6"  # pinned, 2026-10-06
    device: str = "cuda"
    # policy camera slot -> MuJoCo camera name
    camera_map: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_CAMERA_MAP))
    image_size: tuple[int, int] = (256, 256)  # (width, height); the checkpoint's input_features shape
    # "as_shipped": LeRobot's pipeline unchanged (stats do not apply, see module doc).
    # "checkpoint_action_stats": apply the checkpoint's own saved action stats (buffer
    # ``stats_buffer``) to action and, lacking state stats, to the state too.
    stats_mode: str = "as_shipped"
    stats_buffer: str = "so100.buffer.action"
    n_action_steps: int | None = None  # None = checkpoint's (50)

    def __post_init__(self) -> None:
        if self.stats_mode not in STATS_MODES:
            raise ValueError(f"stats_mode must be one of {STATS_MODES}, got {self.stats_mode!r}")


def load_smolvla_config(path, **overrides) -> SmolVLAConfig:
    """YAML (configs/policy/*.yaml) -> SmolVLAConfig; keyword overrides win."""
    import yaml

    from so101_vla.config import resolve_path

    with open(resolve_path(path), encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    raw.update({k: v for k, v in overrides.items() if v is not None})
    if "image_size" in raw:
        raw["image_size"] = tuple(raw["image_size"])
    return SmolVLAConfig(**raw)


def _load_buffer_stats(checkpoint: str, revision: str | None) -> dict[str, np.ndarray]:
    """The raw tensors in the checkpoint's saved normalizer state file."""
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file

    name = "policy_preprocessor_step_5_normalizer_processor.safetensors"
    try:  # cached copy first: no network round trip once downloaded
        path = hf_hub_download(checkpoint, name, revision=revision, local_files_only=True)
    except Exception:  # noqa: BLE001 - not cached yet (LocalEntryNotFoundError) -> download
        path = hf_hub_download(checkpoint, name, revision=revision)
    return load_file(path)


class SmolVLAPolicy:
    """Wraps LeRobot's SmolVLA: LeRobot-format observation in, action chunk (LeRobot units) out."""

    def __init__(self, config: SmolVLAConfig | None = None) -> None:
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as _LeRobotSmolVLA

        self.config = cfg = config or SmolVLAConfig()
        self._torch = torch
        t0 = time.perf_counter()
        self.policy = _LeRobotSmolVLA.from_pretrained(cfg.checkpoint, revision=cfg.revision).to(cfg.device).eval()
        if cfg.n_action_steps is not None:
            self.policy.config.n_action_steps = cfg.n_action_steps
        rename = {f"{IMAGE_PREFIX}{src}": f"{IMAGE_PREFIX}{slot}" for slot, src in self._obs_keys().items()}
        pre_over: dict[str, Any] = {"device_processor": {"device": cfg.device},
                                    "rename_observations_processor": {"rename_map": rename}}
        post_over: dict[str, Any] = {}
        self.stats_used: dict[str, dict[str, list[float]]] = {}
        if cfg.stats_mode == "checkpoint_action_stats":
            raw = _load_buffer_stats(cfg.checkpoint, cfg.revision)
            mean, std = raw[f"{cfg.stats_buffer}.mean"], raw[f"{cfg.stats_buffer}.std"]
            stats = {"action": {"mean": mean, "std": std}, STATE_KEY: {"mean": mean, "std": std}}
            pre_over["normalizer_processor"] = {"stats": stats}
            post_over["unnormalizer_processor"] = {"stats": {"action": stats["action"]}}
            self.stats_used = {k: {s: v.tolist() for s, v in d.items()} for k, d in stats.items()}
        self.pre, self.post = make_pre_post_processors(
            self.policy.config, cfg.checkpoint, pretrained_revision=cfg.revision,
            preprocessor_overrides=pre_over, postprocessor_overrides=post_over)
        self.load_seconds = time.perf_counter() - t0
        self.last_inference_ms: float | None = None

    def _obs_keys(self) -> dict[str, str]:
        """policy slot -> observation key we produce (MuJoCo camera name with '/' made key-safe)."""
        return {slot: cam.replace("/", "_") for slot, cam in self.config.camera_map.items()}

    # --- introspection (Task 2.3) ---------------------------------------------------

    def describe(self) -> dict:
        """What the checkpoint expects and how this wrapper feeds it."""
        pc = self.policy.config
        loaded = {}
        for name, pipe in (("preprocessor", self.pre), ("postprocessor", self.post)):
            for st in pipe.steps:
                if hasattr(st, "_tensor_stats"):
                    loaded[f"{name}.{type(st).__name__}"] = sorted(st._tensor_stats)
        return {
            "checkpoint": self.config.checkpoint, "revision": self.config.revision,
            "input_features": {k: {"type": v.type.value, "shape": list(v.shape)} for k, v in pc.input_features.items()},
            "output_features": {k: {"type": v.type.value, "shape": list(v.shape)} for k, v in pc.output_features.items()},
            "chunk_size": pc.chunk_size, "n_action_steps": pc.n_action_steps,
            "normalization_mapping": {k: v.value for k, v in pc.normalization_mapping.items()},
            "resize_imgs_with_padding": list(pc.resize_imgs_with_padding) if pc.resize_imgs_with_padding else None,
            "max_state_dim": pc.max_state_dim, "max_action_dim": pc.max_action_dim,
            "num_flow_steps": pc.num_steps, "vlm": pc.vlm_model_name, "device": self.config.device,
            "camera_map": self.config.camera_map, "image_size": list(self.config.image_size),
            "stats_mode": self.config.stats_mode, "normalizer_stats_keys_loaded": loaded,
            "stats_applied_to": sorted(self.stats_used) or "none (keys do not match 'action'/'observation.state')",
            "load_seconds": round(self.load_seconds, 1),
        }

    @property
    def camera_names(self) -> list[str]:
        """MuJoCo cameras to render, in policy slot order."""
        return [self.config.camera_map[s] for s in sorted(self.config.camera_map)]

    # --- inference -------------------------------------------------------------------

    def reset(self) -> None:
        self.policy.reset()

    def predict_chunk(self, observation: dict, instruction: str) -> np.ndarray:
        """One action chunk (n_action_steps, 6) in LeRobot units, joint order ``LEROBOT_FEATURE_NAMES``.

        ``observation`` is ``RobotInterface.get_observation(cameras=self.camera_names, ...)``:
        ``"<joint>.pos"`` floats plus ``{camera: (H, W, 3) uint8}``. Raw model output after
        LeRobot's postprocessor; it is NOT validated here (that is the action adapter's job).
        """
        from lerobot.envs.utils import preprocess_observation

        torch = self._torch
        state = np.array([observation[k] for k in LEROBOT_FEATURE_NAMES], np.float32)
        pixels = {key: np.asarray(observation[cam])[None] for cam, key in
                  ((c, c.replace("/", "_")) for c in self.config.camera_map.values())}
        batch = preprocess_observation({"pixels": pixels, "agent_pos": state[None]})
        batch["task"] = [instruction]
        batch = self.pre(batch)
        if self.config.device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            chunk = self.policy.predict_action_chunk(batch)
            chunk = self.post(chunk)
        if self.config.device.startswith("cuda"):
            torch.cuda.synchronize()
        self.last_inference_ms = (time.perf_counter() - t0) * 1000
        out = chunk.detach().float().cpu().numpy()[0]
        return out[: self.policy.config.n_action_steps]
