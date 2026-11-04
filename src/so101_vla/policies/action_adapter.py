"""Check a predicted action chunk before any of it reaches the robot (Tasks 2.4, 2.5).

Raw model output is never treated as a safe joint target. A chunk passes through:

1. here: shape (H, 6), finite values, range and jump checks in LeRobot units,
   giving a report. Malformed chunks (wrong shape, NaN/inf) are rejected.
2. ``CommandSelector``: LeRobot units -> radians (``conventions.LeRobotUnits``),
   safe engage on the first action.
3. ``ArmController``'s ``SafetyFilter``: soft-limit clipping, velocity/acceleration
   limits (counted in the tracking log), then the backend's hard-limit check.

So out-of-range actions are not silently "fixed" here; they are reported here and
clipped (and counted) by the safety filter, like any other command source.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from so101_vla.conventions import JOINT_NAMES, NUM_JOINTS, LeRobotUnits


class InvalidChunk(ValueError):
    """A chunk that must not be executed at all."""


@dataclass(frozen=True)
class ChunkReport:
    steps: int
    finite: bool
    out_of_range_steps: dict[str, int]  # per joint, outside the hard limits
    max_excess: dict[str, float]  # per joint, beyond the hard limit [LeRobot units]
    first_step_distance: dict[str, float]  # |chunk[0] - current| [LeRobot units]
    max_step_change: dict[str, float]  # max |chunk[k+1] - chunk[k]| [LeRobot units]
    min: dict[str, float]
    max: dict[str, float]

    @property
    def any_out_of_range(self) -> bool:
        return any(self.out_of_range_steps.values())

    def to_dict(self) -> dict:
        return {**asdict(self), "any_out_of_range": self.any_out_of_range}


def _per_joint(values: np.ndarray, digits: int = 3) -> dict[str, float]:
    return dict(zip(JOINT_NAMES, np.round(values.astype(float), digits).tolist(), strict=True))


class ActionChunkValidator:
    def __init__(self, units: LeRobotUnits) -> None:
        self.units = units
        self.lower_lr = units.to_lerobot(units.lower)
        self.upper_lr = units.to_lerobot(units.upper)

    def validate(self, chunk: np.ndarray, current_q: np.ndarray) -> ChunkReport:
        """Report on ``chunk`` (H, 6) in LeRobot units; raise InvalidChunk if it must not run."""
        a = np.asarray(chunk, dtype=np.float64)
        if a.ndim != 2 or a.shape[1] != NUM_JOINTS or a.shape[0] < 1:
            raise InvalidChunk(f"expected an action chunk of shape (H, {NUM_JOINTS}), got {a.shape}")
        if not np.all(np.isfinite(a)):
            bad = np.argwhere(~np.isfinite(a))[:5].tolist()
            raise InvalidChunk(f"action chunk contains NaN/inf at (step, joint) {bad}")
        below, above = a < self.lower_lr - 1e-6, a > self.upper_lr + 1e-6
        excess = np.maximum(self.lower_lr - a, a - self.upper_lr).clip(min=0).max(axis=0)
        current = self.units.to_lerobot(np.asarray(current_q, float))
        step_change = np.abs(np.diff(a, axis=0)).max(axis=0) if len(a) > 1 else np.zeros(NUM_JOINTS)
        return ChunkReport(
            steps=len(a), finite=True,
            out_of_range_steps=dict(zip(JOINT_NAMES, (below | above).sum(axis=0).tolist(), strict=True)),
            max_excess=_per_joint(excess), first_step_distance=_per_joint(np.abs(a[0] - current)),
            max_step_change=_per_joint(step_change), min=_per_joint(a.min(axis=0)), max=_per_joint(a.max(axis=0)),
        )
