"""Joint-space trajectories (numpy only)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# Minimum-jerk profile s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5 on tau in [0, 1]:
# peak velocity = 1.875 * dq / T, peak acceleration = (10 / sqrt(3)) * dq / T^2.
_MJ_PEAK_VEL = 1.875
_MJ_PEAK_ACC = 10.0 / np.sqrt(3.0)


@dataclass(frozen=True)
class Trajectory:
    """Joint positions at increasing times; linearly interpolated between samples."""

    times: np.ndarray  # (N,) seconds, times[0] == 0
    positions: np.ndarray  # (N, n_joints) rad

    def __post_init__(self) -> None:
        if self.times.ndim != 1 or self.positions.ndim != 2 or len(self.times) != len(self.positions):
            raise ValueError("times must be (N,) and positions (N, n_joints)")
        if len(self.times) == 0 or self.times[0] != 0.0 or np.any(np.diff(self.times) <= 0):
            raise ValueError("times must start at 0 and be strictly increasing")
        if not np.all(np.isfinite(self.positions)):
            raise ValueError("trajectory contains NaN/inf")

    @property
    def duration(self) -> float:
        return float(self.times[-1])

    @property
    def start(self) -> np.ndarray:
        return self.positions[0].copy()

    @property
    def end(self) -> np.ndarray:
        return self.positions[-1].copy()

    def sample(self, t: float) -> np.ndarray:
        """Position at time ``t`` (held constant outside [0, duration])."""
        return np.array([np.interp(t, self.times, self.positions[:, j]) for j in range(self.positions.shape[1])])

    def then(self, other: Trajectory) -> Trajectory:
        """Concatenate ``other`` after this trajectory (other must start where this ends)."""
        if not np.allclose(other.start, self.end, atol=1e-6):
            raise ValueError("trajectories are not continuous")
        times = np.concatenate([self.times, self.duration + other.times[1:]])
        return Trajectory(times, np.vstack([self.positions, other.positions[1:]]))


def min_jerk_duration(
    q0: np.ndarray, q1: np.ndarray, max_velocity: np.ndarray, max_acceleration: np.ndarray, min_duration: float = 0.0
) -> float:
    """Shortest duration for which a min-jerk move q0 -> q1 respects both limits."""
    dq = np.abs(np.asarray(q1, float) - np.asarray(q0, float))
    t_vel = np.max(_MJ_PEAK_VEL * dq / max_velocity)
    t_acc = np.max(np.sqrt(_MJ_PEAK_ACC * dq / max_acceleration))
    return float(max(t_vel, t_acc, min_duration))


def min_jerk(q0: np.ndarray, q1: np.ndarray, duration: float, dt: float) -> Trajectory:
    """Smooth point-to-point move with zero velocity and acceleration at both ends."""
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    if duration <= 0:
        return Trajectory(np.array([0.0, dt]), np.vstack([q0, q1]))
    n = max(1, int(np.ceil(duration / dt - 1e-9)))
    times = np.linspace(0.0, duration, n + 1)
    tau = times / duration
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    return Trajectory(times, q0 + s[:, None] * (q1 - q0))


def waypoint_trajectory(
    start: np.ndarray,
    waypoints: Sequence[np.ndarray],
    max_velocity: np.ndarray,
    max_acceleration: np.ndarray,
    dt: float,
    speed_scale: float = 1.0,
    min_segment: float = 0.2,
) -> Trajectory:
    """Min-jerk segments start -> waypoint[0] -> ... stopping briefly at each waypoint.

    ``speed_scale`` in (0, 1] slows every segment down relative to the limits.
    """
    if not 0 < speed_scale <= 1:
        raise ValueError("speed_scale must be in (0, 1]")
    vmax = np.asarray(max_velocity, float) * speed_scale
    amax = np.asarray(max_acceleration, float) * speed_scale**2
    traj = Trajectory(np.array([0.0]), np.asarray(start, float)[None, :])
    for wp in waypoints:
        seg_t = min_jerk_duration(traj.end, wp, vmax, amax, min_segment)
        traj = traj.then(min_jerk(traj.end, wp, seg_t, dt))
    return traj
