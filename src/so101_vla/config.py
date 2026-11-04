"""YAML configuration loading.

Relative paths in config files are resolved against the project root (the
directory containing pyproject.toml).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from so101_vla.conventions import NUM_JOINTS

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _vector(value: float | list[float], name: str) -> tuple[float, ...]:
    """Accept a scalar (broadcast to all joints) or a per-joint list."""
    if isinstance(value, (int, float)):
        return (float(value),) * NUM_JOINTS
    if len(value) != NUM_JOINTS:
        raise ValueError(f"{name}: expected {NUM_JOINTS} values, got {len(value)}")
    return tuple(float(v) for v in value)


@dataclass(frozen=True)
class ArmConfig:
    name: str
    model: Path
    mount_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mount_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # w x y z


@dataclass(frozen=True)
class SafetyConfig:
    # Soft limits are the model's hard limits shrunk by this margin [rad], per
    # joint. The gripper defaults to 0 so its full range (as used by LeRobot's
    # 0..100 mapping) stays reachable; pressing on its stop is harmless.
    position_margin: tuple[float, ...] = (0.02,) * (NUM_JOINTS - 1) + (0.0,)
    max_velocity: tuple[float, ...] = (1.5,) * NUM_JOINTS  # rad/s
    max_acceleration: tuple[float, ...] = (6.0,) * NUM_JOINTS  # rad/s^2
    # Max |target - measured position| per joint [rad], as LeRobot's follower
    # max_relative_target (there in degrees / 0..100). None = off (LeRobot's default).
    max_relative_target: tuple[float, ...] | None = None
    # "clip": clamp out-of-range targets and record it; "reject": raise.
    on_out_of_range: str = "clip"
    # Log a warning when |target - measured| exceeds this [rad].
    tracking_error_warn: float = 0.15

    def __post_init__(self) -> None:
        if self.on_out_of_range not in ("clip", "reject"):
            raise ValueError(f"on_out_of_range must be 'clip' or 'reject', got {self.on_out_of_range!r}")


@dataclass(frozen=True)
class RobotConfig:
    scene: Path
    arms: tuple[ArmConfig, ...]
    # None = use the arm model's timestep (Menagerie: 0.005 s). Either way it
    # is nudged so a control period is a whole number of physics steps.
    physics_timestep: float | None = None
    control_hz: float = 30.0  # LeRobot SO-101 standard rate
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    # Named joint-space poses [rad], applied per arm.
    poses: dict[str, tuple[float, ...]] = field(default_factory=dict)

    @property
    def arm_names(self) -> tuple[str, ...]:
        return tuple(a.name for a in self.arms)


def _parse_robot_config(raw: dict[str, Any]) -> RobotConfig:
    arms = []
    for a in raw["arms"]:
        mount = a.get("mount", {})
        arms.append(
            ArmConfig(
                name=a["name"],
                model=resolve_path(a["model"]),
                mount_pos=tuple(mount.get("pos", (0.0, 0.0, 0.0))),
                mount_quat=tuple(mount.get("quat", (1.0, 0.0, 0.0, 0.0))),
            )
        )
    names = [a.name for a in arms]
    if len(set(names)) != len(names):
        raise ValueError(f"arm names must be unique, got {names}")

    s = raw.get("safety", {})
    defaults = SafetyConfig()
    safety = SafetyConfig(
        position_margin=_vector(s.get("position_margin", list(defaults.position_margin)), "position_margin"),
        max_velocity=_vector(s.get("max_velocity", list(defaults.max_velocity)), "max_velocity"),
        max_acceleration=_vector(s.get("max_acceleration", list(defaults.max_acceleration)), "max_acceleration"),
        max_relative_target=(None if s.get("max_relative_target") is None
                             else _vector(s["max_relative_target"], "max_relative_target")),
        on_out_of_range=s.get("on_out_of_range", defaults.on_out_of_range),
        tracking_error_warn=float(s.get("tracking_error_warn", defaults.tracking_error_warn)),
    )
    poses = {k: _vector(v, f"poses.{k}") for k, v in raw.get("poses", {}).items()}

    return RobotConfig(
        scene=resolve_path(raw["scene"]),
        arms=tuple(arms),
        physics_timestep=None if raw.get("physics_timestep") is None else float(raw["physics_timestep"]),
        control_hz=float(raw.get("control_hz", 30.0)),
        safety=safety,
        poses=poses,
    )


def load_robot_config(path: str | Path) -> RobotConfig:
    with open(resolve_path(path), encoding="utf-8") as f:
        return _parse_robot_config(yaml.safe_load(f))
