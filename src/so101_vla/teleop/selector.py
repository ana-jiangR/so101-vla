"""Command selection: exactly one source drives the follower at a time.

    sources (virtual/real leader, VLA policy) -- LeRobot-style get_action()
        -> CommandSelector (one active source; safe engage on every switch)
        -> joint action adapter (conventions.LeRobotUnits: degrees or range_m100_100)
        -> ArmController (safety filter) -> RobotInterface (MuJoCo / real)

Safe engage: when a source is selected, the follower does not jump to that
source's pose. It approaches it at a reduced speed (``engage_velocity``) and
only switches to full-speed tracking once every joint is within
``engage_tolerance``. This works with a moving leader (the target is re-read
every tick) and protects both the initial alignment and later switches
between teleoperation and a policy.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from so101_vla.control.controller import ArmController, StepReport
from so101_vla.conventions import JOINT_NAMES, LeRobotUnits, lerobot_feature_names
from so101_vla.teleop.leaders import CommandSource

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EngageConfig:
    # rad/s (fastest joint) while aligning to a newly selected source. Must
    # exceed how fast the source moves while aligning, or it is never reached.
    engage_velocity: float = 1.0
    engage_tolerance: float = 0.03  # rad; all joints within this -> engaged


class TeleopLog:
    def __init__(self) -> None:
        # time = sim time when the source was read; "measured" is the follower
        # position one control period later (after the physics step).
        self.time: list[float] = []
        self.source: list[str] = []
        self.engaged: list[bool] = []
        self.leader_lerobot: list[np.ndarray] = []  # raw source action (LeRobot units)
        self.leader: list[np.ndarray] = []  # source action in rad
        self.commanded: list[np.ndarray] = []  # after engage limiting + safety filter
        self.measured: list[np.ndarray] = []
        self.compute_ms: list[float] = []  # wall time for read + map + filter + step (no viewer/pacing)
        self.events: list[dict] = []

    def __len__(self) -> int:
        return len(self.time)

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "time": np.asarray(self.time), "source": np.asarray(self.source), "engaged": np.asarray(self.engaged),
            "leader_lerobot": np.asarray(self.leader_lerobot), "leader": np.asarray(self.leader),
            "commanded": np.asarray(self.commanded), "measured": np.asarray(self.measured),
            "compute_ms": np.asarray(self.compute_ms), "joint_names": np.asarray(JOINT_NAMES),
        }

    def summary(self, dt: float) -> dict:
        if not self.time:
            return {"ticks": 0}
        a = self.arrays()
        engaged = a["engaged"]
        out: dict = {"ticks": len(self), "duration_s": round(float(a["time"][-1] - a["time"][0] + dt), 3),
                     "engaged_ticks": int(engaged.sum()), "events": self.events}
        if engaged.any():
            # Error between what the leader commanded and where the follower is
            # one control period later (includes the 1-tick command latency).
            err = (a["leader"] - a["measured"])[engaged]
            out["leader_to_follower_error_rad"] = {
                "max_abs": dict(zip(JOINT_NAMES, np.abs(err).max(0).round(4).tolist(), strict=True)),
                "rms": dict(zip(JOINT_NAMES, np.sqrt((err**2).mean(0)).round(4).tolist(), strict=True)),
            }
            out["follower_lag_ms"] = estimate_lag_ms(a["leader"], a["measured"], dt, engaged)
            out["follower_lag_note"] = ("delay from leader sample to matching follower position, "
                                        "including one control period of zero-order hold")
        c = a["compute_ms"]
        out["compute_latency_ms"] = {"mean": round(float(c.mean()), 3), "p95": round(float(np.percentile(c, 95)), 3),
                                     "max": round(float(c.max()), 3), "budget": round(dt * 1000, 3)}
        return out

    def save(self, directory: str | Path, dt: float) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / "teleop.npz", **self.arrays())
        (directory / "teleop_summary.json").write_text(json.dumps(self.summary(dt), indent=2), encoding="utf-8")
        return directory


def estimate_lag_ms(leader: np.ndarray, measured: np.ndarray, dt: float, mask: np.ndarray | None = None,
                    max_lag_s: float = 0.5) -> dict[str, float | None]:
    """Per-joint delay of the follower behind the leader [ms].

    ``leader[i]`` is the command read at tick i; ``measured[i]`` is the
    follower position at the end of tick i, i.e. one period ``dt`` later.
    Finds the shift k that best aligns ``leader[i]`` with ``measured[i + k]``
    (least squares over ticks where ``mask`` holds at both ends, sub-tick via
    parabolic interpolation); lag = (k + 1) * dt. Joints that barely moved
    return None.
    """
    n = len(leader)
    mask = np.ones(n, bool) if mask is None else np.asarray(mask, bool)
    out: dict[str, float | None] = {}
    max_k = min(int(max_lag_s / dt), n // 2)
    for j, name in enumerate(JOINT_NAMES):
        x, y = leader[:, j], measured[:, j]
        if np.ptp(x[mask]) < 0.05 or max_k < 2:
            out[name] = None
            continue
        cost = np.array([np.mean((x[: n - k] - y[k:])[mask[: n - k] & mask[k:]] ** 2) for k in range(max_k + 1)])
        k = int(cost.argmin())
        frac = 0.0
        if 0 < k < max_k:
            c0, c1, c2 = cost[k - 1], cost[k], cost[k + 1]
            denom = c0 - 2 * c1 + c2
            frac = 0.5 * (c0 - c2) / denom if denom > 0 else 0.0
        out[name] = round((k + frac + 1) * dt * 1000, 1)
    return out


class CommandSelector:
    """Routes exactly one command source to one follower arm per control tick."""

    def __init__(self, controller: ArmController, arm: str, sources: Mapping[str, CommandSource],
                 engage: EngageConfig | None = None, key_prefix: str = "",
                 on_source_error: str = "hold", norm: str = "degrees") -> None:
        """
        key_prefix:       prefix of this arm's keys in the source's actions, e.g.
                          ``"left_"`` for LeRobot's bimanual ``bi_so_leader``.
        on_source_error:  "hold" (default): if the active source raises (e.g. a
                          serial read fails), log it, deselect the source, and hold
                          position; "raise": propagate the exception.
        norm:             LeRobot unit mode of the sources' arm joints: "degrees"
                          (use_degrees=True, LeRobot's sim env) or "range_m100_100"
                          (LeRobot's default). The gripper is always 0..100.
        """
        if arm not in controller.arm_names:
            raise ValueError(f"unknown arm {arm!r}")
        if on_source_error not in ("hold", "raise"):
            raise ValueError("on_source_error must be 'hold' or 'raise'")
        self.controller = controller
        self.arm = arm
        self.sources = dict(sources)
        self.engage = engage or EngageConfig()
        self.key_prefix = key_prefix
        self.on_source_error = on_source_error
        self.active: str | None = None
        self.engaged = False
        self.log = TeleopLog()
        lim = controller.robot.joint_limits(arm)
        self.units = LeRobotUnits(lim.lower, lim.upper, norm)

    def select(self, name: str | None) -> None:
        """Make ``name`` the only active source (None = hold position). Re-engages safely."""
        if name is not None and name not in self.sources:
            raise KeyError(f"unknown source {name!r}; have {sorted(self.sources)}")
        if name != self.active:
            self.log.events.append({"t": round(self.controller.get_state().time, 4), "select": name})
        self.active = name
        self.engaged = False

    def to_lerobot(self, q: np.ndarray) -> np.ndarray:
        return self.units.to_lerobot(q)

    def tick(self) -> StepReport:
        """Read the active source, map, limit, and step the follower once."""
        t0 = time.perf_counter()
        t_sample = self.controller.get_state().time
        current = self.controller.filters[self.arm].last_target
        action = None
        if self.active is not None:
            try:
                action = self.sources[self.active].get_action()
                leader_q = self.units.from_dict(action, self.key_prefix)
            except Exception as exc:
                if self.on_source_error == "raise":
                    raise
                log.error("source %r failed (%s: %s); holding position", self.active, type(exc).__name__, exc)
                self.log.events.append({"t": round(t_sample, 4), "source_error": self.active,
                                        "error": f"{type(exc).__name__}: {exc}"})
                self.select(None)
                action = None
        if action is None:  # no active source (or it just failed): hold
            raw = self.to_lerobot(current)
            leader_q = current
            target = current
        else:
            raw = np.array([action[k] for k in lerobot_feature_names(self.key_prefix)])
            target = leader_q
            if not self.engaged:
                # Move all joints proportionally (straight line in joint space,
                # like move_to) with the slowest-allowed joint speed capped.
                delta = leader_q - current
                step = self.engage.engage_velocity * self.controller.dt
                target = current + delta * min(1.0, step / max(np.abs(delta).max(), 1e-12))
                if np.abs(delta).max() <= self.engage.engage_tolerance:
                    self.engaged = True
                    self.log.events.append({"t": round(self.controller.get_state().time, 4), "engaged": self.active})
                    target = leader_q
        t_cmd = time.perf_counter()
        report = self.controller.command({self.arm: target})
        # read + map + engage (ours) + filter + step (controller); viewer/pacing excluded
        compute_ms = ((t_cmd - t0) + self.controller.last_compute_s) * 1000

        tl = self.log
        tl.time.append(t_sample)
        tl.source.append(self.active or "")
        tl.engaged.append(self.engaged)
        tl.leader_lerobot.append(raw)
        tl.leader.append(leader_q)
        tl.commanded.append(report.filtered[self.arm].target)
        tl.measured.append(report.state.arms[self.arm].positions)
        tl.compute_ms.append(compute_ms)
        return report

    def run(self, duration: float) -> None:
        for _ in range(max(1, round(duration / self.controller.dt))):
            self.tick()
