"""Zero-shot MolmoAct2-SO100/101 (Ai2, 5.4B) through LeRobot's own APIs, quantized for 8 GB GPUs.

Sources for each step:

* checkpoint: ``lerobot/MolmoAct2-SO100_101-LeRobot`` (LeRobot v0.6.0 release notes): the
  LeRobot config + saved pre/post-processors for ``allenai/MolmoAct2-SO100_101``, with the
  joint-frame correction for arms calibrated with LeRobot >= 0.5 (v3.0) built in
  (``joint_signs`` / ``joint_offsets``: shoulder_lift sign flip, +90 deg on shoulder_lift
  and elbow_flex; ``MolmoAct2StateFrameTransformStep`` / ``MolmoAct2ActionFrameTransformStep``).
* weights: LeRobot's ``MolmoAct2Policy.__init__`` loads them from the config's
  ``checkpoint_path`` (``allenai/MolmoAct2-SO100_101``). We construct the policy directly
  instead of ``from_pretrained``, which would then download the LeRobot repo's own 22 GB
  ``model.safetensors`` as well. Checked 2026-11-11: both repos hold the same 1295 tensors
  (same shapes; sampled tensors byte-identical; LeRobot's keys only add a ``model.`` prefix).
* pre/post-processing: the saved pipelines, via ``make_pre_post_processors`` (as
  ``lerobot-rollout`` does), with only the device and the weights revision overridden.
* observation: ``preprocess_observation`` as in ``lerobot-eval``; images go to
  ``observation.images.cam0`` / ``cam1``. Camera order does not matter for this checkpoint
  (model card: trained with randomized camera order); Ai2 evaluated wrist + third-person.

Quantization is our addition (not a LeRobot feature): bitsandbytes LLM.int8 (``int8``) or
NF4 (``nf4``) replaces ``nn.Linear`` layers after loading, before moving to the GPU. bf16
needs about 11 GB of weights; int8 on the LLM + ViT brings that to about 6 GB.
Kept in bf16: the flow-matching action expert (0.57B, precision-sensitive), embeddings,
norms, the ViT patch embedding (its weight dtype sets the input dtype) and the image
projector. The continuous action path never calls ``lm_head`` (0.4B; it runs
``MolmoAct2Model`` only, ``generate_actions_from_inputs``), so by default it stays on the
CPU (``offload_lm_head``); measured 2026-11-11 on an 8 GB RTX 4060 Laptop, int8 with it on
the GPU filled the card (8.59/8.59 GB). Discrete / depth-reasoning modes would need it back.
CUDA-graph capture is off by default: it costs about 2 GB and is untested with bitsandbytes.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from so101_vla.conventions import LEROBOT_FEATURE_NAMES

QUANTIZATIONS = ("none", "int8", "nf4")

# Prefixes are relative to the HF model (``MolmoAct2ForConditionalGeneration``).
DEFAULT_QUANTIZE_PREFIXES = ("model.transformer.blocks.", "model.vision_backbone.image_vit.")
NEVER_QUANTIZE = ("patch_embedding", "action_expert")


@dataclass(frozen=True)
class MolmoAct2Config:
    checkpoint: str = "lerobot/MolmoAct2-SO100_101-LeRobot"
    revision: str | None = "a4ab68081b84bb0659d96f454edbce6abbe8a1cd"  # pinned, 2026-11-11
    # Revision of the weights repo the config points to (allenai/MolmoAct2-SO100_101).
    weights_revision: str | None = "152569fe57914d97be91055800035f54e250d009"  # pinned, 2026-11-11
    device: str = "cuda"
    quantization: str = "int8"
    quantize_prefixes: tuple[str, ...] = DEFAULT_QUANTIZE_PREFIXES
    # policy camera slot -> robot camera name
    camera_map: dict[str, str] = field(default_factory=lambda: {"cam0": "front", "cam1": "main/wrist_cam"})
    image_size: tuple[int, int] = (640, 480)  # (width, height); the processor resizes to 378x378
    n_action_steps: int | None = None  # None = checkpoint's 30 (1 s at 30 Hz)
    num_steps: int | None = None  # flow-matching steps; None = checkpoint's 10
    cuda_graph: bool = False
    offload_lm_head: bool = True  # unused by continuous inference; keep its 0.4B params off the GPU
    # Local bf16 copy of the weights repo (made once, ~11 GB). The Hub shards are fp32 and LeRobot
    # reads every shard twice (HF from_pretrained, then its strict reload), so loading the fp32
    # repo took 11-12 min here. None = load the Hub repo directly.
    bf16_cache: str | None = "outputs/models/MolmoAct2-SO100_101-bf16"

    def __post_init__(self) -> None:
        if self.quantization not in QUANTIZATIONS:
            raise ValueError(f"quantization must be one of {QUANTIZATIONS}, got {self.quantization!r}")
        if self.quantization != "none" and not self.device.startswith("cuda"):
            raise ValueError("bitsandbytes quantization needs a CUDA device; use quantization: none on CPU")


def load_molmoact2_config(path, **overrides) -> MolmoAct2Config:
    """YAML (configs/policy/*.yaml) -> MolmoAct2Config; keyword overrides that are not None win."""
    import yaml

    from so101_vla.config import resolve_path

    with open(resolve_path(path), encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    raw.pop("type", None)
    raw.update({k: v for k, v in overrides.items() if v is not None})
    for key in ("image_size", "quantize_prefixes"):
        if key in raw:
            raw[key] = tuple(raw[key])
    return MolmoAct2Config(**raw)


def ensure_bf16_cache(repo: str, revision: str | None, cache_dir) -> str:
    """Local copy of the HF weights repo with float tensors stored as bf16 (same keys and shards).

    The same fp32 -> bf16 cast LeRobot's loader does at ``model_dtype="bfloat16"``, done once on
    disk. Non-weight files (config, norm_stats.json, processor, tokenizer, remote code) are copied
    unchanged. ``SOURCE.txt`` is written last, so an interrupted conversion is redone.
    """
    import json
    import shutil
    from pathlib import Path

    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file

    from so101_vla.config import resolve_path

    out = Path(resolve_path(cache_dir))
    source = f"{repo}@{revision}\n"
    marker = out / "SOURCE.txt"
    if marker.exists() and marker.read_text(encoding="utf-8") == source:
        return str(out)
    src = Path(snapshot_download(repo, revision=revision))
    out.mkdir(parents=True, exist_ok=True)
    marker.unlink(missing_ok=True)
    for f in src.rglob("*"):
        if f.is_file() and f.suffix != ".safetensors" and f.name != "model.safetensors.index.json":
            (out / f.relative_to(src)).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, out / f.relative_to(src))
    total = 0
    for shard in sorted(src.glob("*.safetensors")):
        print(f"bf16 cache: converting {shard.name} ...", flush=True)
        tensors = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v).contiguous()
                   for k, v in load_file(shard).items()}
        total += sum(v.numel() * v.element_size() for v in tensors.values())
        save_file(tensors, out / shard.name, metadata={"format": "pt"})
        del tensors
    index = src / "model.safetensors.index.json"
    if index.exists():
        idx = json.loads(index.read_text(encoding="utf-8"))
        idx.setdefault("metadata", {})["total_size"] = total
        (out / index.name).write_text(json.dumps(idx, indent=2), encoding="utf-8")
    marker.write_text(source, encoding="utf-8")
    return str(out)


def _quantize_linears(model, prefixes: tuple[str, ...], mode: str) -> dict[str, int]:
    """Replace matching ``nn.Linear`` layers with bitsandbytes ones (weights quantize on ``.to(cuda)``)."""
    import bitsandbytes as bnb
    import torch
    from torch import nn

    targets = [(name, mod) for name, mod in model.named_modules()
               if type(mod) is nn.Linear and name.startswith(prefixes)
               and not any(tok in name for tok in NEVER_QUANTIZE)]
    params = 0
    for name, lin in targets:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        has_bias = lin.bias is not None
        if mode == "int8":
            new = bnb.nn.Linear8bitLt(lin.in_features, lin.out_features, bias=has_bias,
                                      has_fp16_weights=False, threshold=6.0)
            new.weight = bnb.nn.Int8Params(lin.weight.data.to(torch.float16), requires_grad=False,
                                           has_fp16_weights=False)
        else:
            new = bnb.nn.Linear4bit(lin.in_features, lin.out_features, bias=has_bias,
                                    compute_dtype=torch.bfloat16, quant_type="nf4")
            new.weight = bnb.nn.Params4bit(lin.weight.data.to(torch.bfloat16), requires_grad=False,
                                           quant_type="nf4")
        if has_bias:
            new.bias = nn.Parameter(lin.bias.data, requires_grad=False)
        setattr(parent, child, new)
        params += lin.in_features * lin.out_features
    return {"layers": len(targets), "params": params}


class MolmoAct2Policy:
    """Wraps LeRobot's MolmoAct2: LeRobot-format observation in, action chunk (LeRobot degrees) out.

    Same ``ChunkPolicy`` contract as ``SmolVLAPolicy``, so it plugs into ``PolicySource``.
    """

    def __init__(self, config: MolmoAct2Config | None = None) -> None:
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy as _LeRobotMolmoAct2

        self.config = cfg = config or MolmoAct2Config()
        self._torch = torch
        # transformers 5's threaded safetensors loader (core_model_loading, 4 workers) segfaulted
        # in _materialize_copy while converting the 22 GB fp32 shards to bf16 on Windows
        # (2026-11-11, 1 of 2 loads). Its documented switch makes loading sequential.
        os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
        t0 = time.perf_counter()
        pc = PreTrainedConfig.from_pretrained(cfg.checkpoint, revision=cfg.revision)
        self.weights_source = f"{pc.checkpoint_path}@{cfg.weights_revision}"
        pc.checkpoint_revision = cfg.weights_revision
        if cfg.bf16_cache:
            pc.checkpoint_path = ensure_bf16_cache(pc.checkpoint_path, cfg.weights_revision, cfg.bf16_cache)
            pc.checkpoint_revision = None
        pc.model_dtype = "bfloat16"  # model card: bf16 "usually does not hurt performance much"
        pc.device = cfg.device
        pc.enable_inference_cuda_graph = cfg.cuda_graph
        if cfg.n_action_steps is not None:
            pc.n_action_steps = cfg.n_action_steps
        if cfg.num_steps is not None:
            pc.num_inference_steps = cfg.num_steps
        self.policy = _LeRobotMolmoAct2(pc).eval()  # built on CPU
        self.quantized = {"layers": 0, "params": 0}
        if cfg.quantization != "none":
            self.quantized = _quantize_linears(self.policy._hf_model(), cfg.quantize_prefixes, cfg.quantization)
        if cfg.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        hf = self.policy._hf_model()
        lm_head = hf.lm_head if cfg.offload_lm_head else None
        if lm_head is not None:
            hf.lm_head = None  # detach so .to(device) skips it, then reattach on the CPU
        self.policy.to(cfg.device)
        if lm_head is not None:
            hf.lm_head = lm_head
        pack = {"checkpoint_path": pc.checkpoint_path, "checkpoint_revision": pc.checkpoint_revision}
        self.pre, self.post = make_pre_post_processors(
            pc, cfg.checkpoint, pretrained_revision=cfg.revision,
            preprocessor_overrides={"device_processor": {"device": cfg.device}, "molmoact2_pack_inputs": pack})
        self.load_seconds = time.perf_counter() - t0
        self.last_inference_ms: float | None = None

    # --- introspection ---------------------------------------------------------------

    def gpu_memory_gb(self) -> dict[str, float]:
        torch = self._torch
        if not self.config.device.startswith("cuda"):
            return {}
        free, total = torch.cuda.mem_get_info()
        return {"allocated": round(torch.cuda.memory_allocated() / 1e9, 2),
                "peak_allocated": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                "device_used": round((total - free) / 1e9, 2), "device_total": round(total / 1e9, 2)}

    def describe(self) -> dict:
        pc = self.policy.config
        return {
            "checkpoint": self.config.checkpoint, "revision": self.config.revision,
            "weights": self.weights_source, "weights_loaded_from": pc.checkpoint_path,
            "input_features": {k: {"type": v.type.value, "shape": list(v.shape)} for k, v in pc.input_features.items()},
            "output_features": {k: {"type": v.type.value, "shape": list(v.shape)} for k, v in pc.output_features.items()},
            "chunk_size": pc.chunk_size, "n_action_steps": pc.n_action_steps,
            "num_inference_steps": pc.num_inference_steps, "norm_tag": pc.norm_tag,
            "joint_signs": pc.joint_signs, "joint_offsets": pc.joint_offsets,
            "quantization": self.config.quantization, "quantized_linear_layers": self.quantized["layers"],
            "quantized_params_billion": round(self.quantized["params"] / 1e9, 3),
            "camera_map": self.config.camera_map, "image_size": list(self.config.image_size),
            "device": self.config.device, "gpu_memory_gb": self.gpu_memory_gb(),
            "load_seconds": round(self.load_seconds, 1),
        }

    @property
    def camera_names(self) -> list[str]:
        """Robot cameras to render, in policy slot order; a camera feeding both slots is listed once."""
        return list(dict.fromkeys(self.config.camera_map[s] for s in sorted(self.config.camera_map)))

    # --- inference -------------------------------------------------------------------

    def reset(self) -> None:
        self.policy.reset()

    def predict_chunk(self, observation: dict, instruction: str) -> np.ndarray:
        """One action chunk (n_action_steps, 6) in LeRobot degrees, joint order ``LEROBOT_FEATURE_NAMES``.

        ``observation`` is ``RobotInterface.get_observation(cameras=self.camera_names, ...)``
        in ``"degrees"`` units. Raw output after LeRobot's postprocessor (including the
        joint-frame correction); validation is the action adapter's job.
        """
        from lerobot.envs.utils import preprocess_observation

        torch = self._torch
        state = np.array([observation[k] for k in LEROBOT_FEATURE_NAMES], np.float32)
        pixels = {slot: np.asarray(observation[cam])[None] for slot, cam in self.config.camera_map.items()}
        batch = preprocess_observation({"pixels": pixels, "agent_pos": state[None]})
        batch["task"] = [instruction]
        batch = self.pre(batch)
        if self.config.device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            chunk = self.post(self.policy.predict_action_chunk(batch))
        if self.config.device.startswith("cuda"):
            torch.cuda.synchronize()
        self.last_inference_ms = (time.perf_counter() - t0) * 1000
        return chunk.detach().float().cpu().numpy()[0][: self.policy.config.n_action_steps]
