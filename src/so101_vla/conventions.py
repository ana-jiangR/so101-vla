"""Canonical joint ordering and unit conventions for one SO-101 follower arm.

Internal convention used everywhere in this package (see docs/conventions.md):

* Joint order: ``JOINT_NAMES`` (same as the MJCF and LeRobot's SO-101
  follower/leader motors).
* Units: radians for all six joints, including the gripper. Zero is the
  "new calibration" zero: the middle of each joint's range.
* Gripper: a hinge on the moving jaw. Lower limit (-0.1745 rad) = closed,
  upper limit (+1.7453 rad) = open.

LeRobot convention (``so101-joint-degrees-v1``, i.e. ``so101_follower`` with
``use_degrees=True``), as implemented by LeRobot's official MuJoCo env
(huggingface.co/lerobot/so101-tabletop-mujoco, ``env.py``):

* the five arm joints in **degrees** (``np.rad2deg`` of the joint angle, no offset)
* the gripper in **0..100**, linear over the actuator ctrlrange (0 = closed)
* feature names ``"<joint>.pos"``

``LeRobotUnits`` also supports LeRobot's default ``range_m100_100`` mode for the
arm joints. All conversions between conventions live in this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
NUM_JOINTS = len(JOINT_NAMES)
ARM_JOINT_NAMES = JOINT_NAMES[:5]
GRIPPER_INDEX = JOINT_NAMES.index("gripper")

# LeRobot observation/action feature names, e.g. "shoulder_pan.pos".
LEROBOT_FEATURE_NAMES: tuple[str, ...] = tuple(f"{j}.pos" for j in JOINT_NAMES)

# Site on the model approximating the tool center point.
EE_SITE = "gripperframe"


def joint_index(name: str) -> int:
    return JOINT_NAMES.index(name)


# --- gripper ------------------------------------------------------------------
# lower/upper are the gripper's hard limits (actuator ctrlrange), the same
# range LeRobot's 0..100 mapping uses.

def gripper_to_normalized(q: float, lower: float, upper: float) -> float:
    """Gripper joint angle [rad] -> opening in [0, 1] (0 = closed, 1 = open)."""
    return float(np.clip((q - lower) / (upper - lower), 0.0, 1.0))


def normalized_to_gripper(opening: float, lower: float, upper: float) -> float:
    """Opening in [0, 1] (0 = closed, 1 = open) -> gripper joint angle [rad]."""
    if not 0.0 <= opening <= 1.0:
        raise ValueError(f"gripper opening must be in [0, 1], got {opening}")
    return float(lower + opening * (upper - lower))


# --- LeRobot units ------------------------------------------------------------

def to_lerobot(q: np.ndarray, gripper_lower: float, gripper_upper: float) -> np.ndarray:
    """Joint angles [rad] -> LeRobot units (5 x degrees, gripper 0..100)."""
    q = np.asarray(q, dtype=np.float64)
    out = np.rad2deg(q)
    out[..., GRIPPER_INDEX] = (q[..., GRIPPER_INDEX] - gripper_lower) / (gripper_upper - gripper_lower) * 100.0
    return out


def from_lerobot(values: np.ndarray, gripper_lower: float, gripper_upper: float) -> np.ndarray:
    """LeRobot units (5 x degrees, gripper 0..100) -> joint angles [rad].

    Unlike LeRobot's env this does not clip: out-of-range values are left for
    the safety layer to clip or reject, so they are visible in its reports.
    """
    values = np.asarray(values, dtype=np.float64)
    out = np.deg2rad(values)
    out[..., GRIPPER_INDEX] = gripper_lower + values[..., GRIPPER_INDEX] / 100.0 * (gripper_upper - gripper_lower)
    return out


# LeRobot MotorNormMode values for the five arm joints (the gripper is always
# range_0_100). "degrees" = so101_follower/leader(use_degrees=True), as used by
# LeRobot's MuJoCo env; "range_m100_100" = LeRobot's default (use_degrees=False),
# which many community SO-100/101 datasets and checkpoints were recorded in.
NORM_MODES: tuple[str, ...] = ("degrees", "range_m100_100")


@dataclass(frozen=True)
class LeRobotUnits:
    """Radians <-> LeRobot units for one arm, in either LeRobot normalization mode.

    ``lower``/``upper`` are the arm's hard limits [rad] (``RobotInterface.joint_limits``).

    * ``degrees``: ``rad2deg(q)``, exactly LeRobot's MuJoCo env.
    * ``range_m100_100``: ``lower..upper`` -> -100..100, linear. On hardware LeRobot maps the
      calibrated ``range_min..range_max`` counts onto -100..100, so this matches real data
      when the calibrated travel spans the model's limits (the convention Strands Robots'
      ``units/lerobot.py`` uses too).
    * gripper: ``lower..upper`` -> 0..100 in both modes (0 = closed).

    Values are not clipped; the safety layer clips or rejects and counts it.
    """

    lower: np.ndarray
    upper: np.ndarray
    norm: str = "degrees"

    def __post_init__(self) -> None:
        if self.norm not in NORM_MODES:
            raise ValueError(f"unknown LeRobot norm mode {self.norm!r}; known: {NORM_MODES}")
        for name in ("lower", "upper"):
            v = np.asarray(getattr(self, name), dtype=np.float64)
            if v.shape != (NUM_JOINTS,):
                raise ValueError(f"{name} must have shape ({NUM_JOINTS},), got {v.shape}")
            object.__setattr__(self, name, v)

    def to_lerobot(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        if self.norm == "degrees":
            out = np.rad2deg(q)
        else:
            out = (q - self.lower) / (self.upper - self.lower) * 200.0 - 100.0
        g = GRIPPER_INDEX
        out[..., g] = (q[..., g] - self.lower[g]) / (self.upper[g] - self.lower[g]) * 100.0
        return out

    def from_lerobot(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if self.norm == "degrees":
            out = np.deg2rad(values)
        else:
            out = self.lower + (values + 100.0) / 200.0 * (self.upper - self.lower)
        g = GRIPPER_INDEX
        out[..., g] = self.lower[g] + values[..., g] / 100.0 * (self.upper[g] - self.lower[g])
        return out

    def to_dict(self, q: np.ndarray, prefix: str = "") -> dict[str, float]:
        return dict(zip(lerobot_feature_names(prefix), self.to_lerobot(q).tolist(), strict=True))

    def from_dict(self, action: dict[str, float], prefix: str = "") -> np.ndarray:
        """``{"<prefix>shoulder_pan.pos": ..., ...}`` -> angles [rad]; extra keys are ignored."""
        names = lerobot_feature_names(prefix)
        missing = [k for k in names if k not in action]
        if missing:
            raise KeyError(f"LeRobot action missing keys: {missing}")
        return self.from_lerobot(np.array([action[k] for k in names]))


def lerobot_feature_names(prefix: str = "") -> tuple[str, ...]:
    """LeRobot keys for one arm. Bimanual LeRobot robots/teleops (bi_so_follower,
    bi_so_leader) prefix each arm's keys with ``"left_"`` / ``"right_"``."""
    return tuple(prefix + k for k in LEROBOT_FEATURE_NAMES)


def to_lerobot_dict(q: np.ndarray, gripper_lower: float, gripper_upper: float, prefix: str = "") -> dict[str, float]:
    """Joint angles [rad] -> ``{"<prefix>shoulder_pan.pos": deg, ..., "<prefix>gripper.pos": 0..100}``."""
    values = to_lerobot(q, gripper_lower, gripper_upper).tolist()
    return dict(zip(lerobot_feature_names(prefix), values, strict=True))


def from_lerobot_dict(action: dict[str, float], gripper_lower: float, gripper_upper: float,
                      prefix: str = "") -> np.ndarray:
    """``{"<prefix>shoulder_pan.pos": ..., ...}`` (LeRobot action/observation) -> angles [rad].

    Extra keys (e.g. the other arm's, or camera entries) are ignored.
    """
    names = lerobot_feature_names(prefix)
    missing = [k for k in names if k not in action]
    if missing:
        raise KeyError(f"LeRobot action missing keys: {missing}")
    return from_lerobot(np.array([action[k] for k in names]), gripper_lower, gripper_upper)
