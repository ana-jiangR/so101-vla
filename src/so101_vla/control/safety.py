"""Per-arm command validation and limiting (numpy only, backend independent).

Every position target goes through ``SafetyFilter`` before reaching a robot
backend. Per control tick it:

1. rejects malformed targets (wrong shape, NaN/inf) -> CommandError
2. clips (or rejects) targets outside the soft position limits
3. limits the commanded velocity and acceleration, using a braking-distance
   bound so the command decelerates and stops at the target without overshoot.
   Targets streamed along a smooth trajectory that already respects the
   limits pass through unchanged.
4. optionally (``max_relative_target``) keeps the target within a fixed
   distance of the *measured* joint position, like LeRobot's
   ``ensure_safe_goal_position`` on the real SO-101 follower.

The output is a target that moves at most ``max_velocity * dt`` per tick, so a
large jump from a policy or a typo becomes a smooth, bounded motion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from so101_vla.config import SafetyConfig
from so101_vla.conventions import JOINT_NAMES
from so101_vla.robot.interface import CommandError, JointLimits, check_target_array


@dataclass(frozen=True)
class SafetyLimits:
    lower: np.ndarray  # soft limits [rad]
    upper: np.ndarray
    max_velocity: np.ndarray  # rad/s
    max_acceleration: np.ndarray  # rad/s^2
    max_relative_target: np.ndarray | None = None  # rad from measured position; None = off

    @classmethod
    def from_config(cls, hard: JointLimits, cfg: SafetyConfig) -> SafetyLimits:
        lower = hard.lower + cfg.position_margin
        upper = hard.upper - cfg.position_margin
        if np.any(lower >= upper):
            raise ValueError("position_margin is larger than half a joint range")
        vmax = np.asarray(cfg.max_velocity, dtype=float)
        amax = np.asarray(cfg.max_acceleration, dtype=float)
        if np.any(vmax <= 0) or np.any(amax <= 0):
            raise ValueError("max_velocity and max_acceleration must be positive")
        mrt = None
        if cfg.max_relative_target is not None:
            mrt = np.asarray(cfg.max_relative_target, dtype=float)
            if np.any(mrt <= 0):
                raise ValueError("max_relative_target must be positive")
        return cls(lower, upper, vmax, amax, mrt)


@dataclass(frozen=True)
class FilterResult:
    target: np.ndarray  # safe target to send this tick
    position_clipped: np.ndarray  # bool per joint: requested target was outside soft limits
    rate_limited: np.ndarray  # bool per joint: velocity/acceleration limit changed the target
    relative_limited: np.ndarray  # bool per joint: max_relative_target (vs. measured) changed it

    @property
    def modified(self) -> bool:
        return bool(self.position_clipped.any() or self.rate_limited.any() or self.relative_limited.any())


class SafetyFilter:
    def __init__(self, limits: SafetyLimits, dt: float, on_out_of_range: str = "clip", name: str = "arm") -> None:
        self.limits = limits
        self.dt = dt
        self.on_out_of_range = on_out_of_range
        self.name = name
        self._q: np.ndarray | None = None  # last output target
        self._v = np.zeros_like(limits.lower)  # last output velocity
        self._goal = np.zeros_like(limits.lower)  # last requested (range-checked) target

    @property
    def last_target(self) -> np.ndarray:
        if self._q is None:
            raise RuntimeError("SafetyFilter.reset() must be called before use")
        return self._q.copy()

    def reset(self, q: np.ndarray) -> None:
        """Start from target ``q`` at rest (call after robot reset/connect)."""
        self._q = np.clip(np.asarray(q, dtype=float), self.limits.lower, self.limits.upper)
        self._v = np.zeros_like(self._q)
        self._goal = self._q.copy()

    def check_range(self, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Validate a target; return (clipped target, out-of-range mask)."""
        q = check_target_array(target, self.name)
        lim = self.limits
        outside = (q < lim.lower - 1e-9) | (q > lim.upper + 1e-9)  # ignore float round-off
        if outside.any() and self.on_out_of_range == "reject":
            bad = [f"{JOINT_NAMES[i]}={q[i]:+.4f} not in [{lim.lower[i]:+.4f}, {lim.upper[i]:+.4f}]"
                   for i in np.flatnonzero(outside)]
            raise CommandError(f"arm {self.name!r}: target outside soft limits: {', '.join(bad)}")
        return np.clip(q, lim.lower, lim.upper), outside

    def __call__(self, target: np.ndarray, measured: np.ndarray | None = None) -> FilterResult:
        """Return the safe target for this tick.

        ``measured`` (current joint positions) enables ``max_relative_target``.
        """
        prev_q = self.last_target
        goal, outside = self.check_range(target)
        lim, dt = self.limits, self.dt

        err = goal - prev_q
        direction = np.sign(err)
        # How fast the requested target itself is moving (a streamed trajectory
        # moves every tick; a single jump counts as one tick at max velocity).
        goal_vel = np.clip((goal - self._goal) / dt, -lim.max_velocity, lim.max_velocity)
        self._goal = goal
        # Fastest approach that can still stop at the target without overshoot:
        # target speed along the error direction + discrete-time braking speed,
        # i.e. the largest v with v*dt + v^2 / (2a) <= |err|.
        a_dt = lim.max_acceleration * dt
        v_brake = -a_dt + np.sqrt(a_dt**2 + 2.0 * lim.max_acceleration * np.abs(err))
        v_stop = np.maximum(direction * goal_vel, 0.0) + v_brake
        v_want = np.clip(err / dt, -lim.max_velocity, lim.max_velocity)
        v_want = direction * np.minimum(np.abs(v_want), v_stop)
        v = np.clip(v_want, self._v - lim.max_acceleration * dt, self._v + lim.max_acceleration * dt)

        q = np.clip(prev_q + v * dt, lim.lower, lim.upper)
        # Snap onto the goal when the remaining error is numerical noise.
        q = np.where(np.abs(goal - q) < 1e-9, goal, q)

        # LeRobot's ensure_safe_goal_position: keep the target within
        # max_relative_target of where the joint actually is, so a blocked
        # joint's target cannot run far ahead and snap when released.
        relative_limited = np.zeros_like(outside)
        if lim.max_relative_target is not None and measured is not None:
            m = np.asarray(measured, dtype=float)
            q_rel = np.clip(q, m - lim.max_relative_target, m + lim.max_relative_target)
            relative_limited = ~np.isclose(q_rel, q, atol=1e-9, rtol=0.0)
            q = np.clip(q_rel, lim.lower, lim.upper)

        self._v = (q - prev_q) / dt
        self._q = q
        rate_limited = ~np.isclose(q, goal, atol=1e-9, rtol=0.0) & ~relative_limited
        return FilterResult(target=q.copy(), position_clipped=outside, rate_limited=rate_limited,
                            relative_limited=relative_limited)
