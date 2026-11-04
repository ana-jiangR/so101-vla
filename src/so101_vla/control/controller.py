"""High-level joint/gripper control on top of any RobotInterface.

``ArmController`` is the single entry point for motion: every command goes
through a per-arm ``SafetyFilter`` before ``robot.set_targets`` and is logged
for tracking-error analysis. It knows nothing about MuJoCo or VLA policies.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from so101_vla.config import RobotConfig, SafetyConfig
from so101_vla.conventions import GRIPPER_INDEX, JOINT_NAMES, joint_index, normalized_to_gripper
from so101_vla.control.safety import FilterResult, SafetyFilter, SafetyLimits
from so101_vla.control.trajectory import Trajectory, min_jerk, min_jerk_duration, waypoint_trajectory
from so101_vla.robot.interface import CommandError, RobotInterface, RobotState

log = logging.getLogger(__name__)

PoseLike = str | np.ndarray | Mapping[str, np.ndarray]
StepCallback = Callable[[RobotState], None]


@dataclass(frozen=True)
class StepReport:
    state: RobotState
    filtered: dict[str, FilterResult]


class TrackingLog:
    """Commanded vs. measured joint positions at every control tick."""

    def __init__(self, arm_names: tuple[str, ...]) -> None:
        self.arm_names = arm_names
        self.times: list[float] = []
        self.commanded: dict[str, list[np.ndarray]] = {a: [] for a in arm_names}
        self.measured: dict[str, list[np.ndarray]] = {a: [] for a in arm_names}
        self.clipped: dict[str, list[np.ndarray]] = {a: [] for a in arm_names}
        self.rate_limited: dict[str, list[np.ndarray]] = {a: [] for a in arm_names}
        self.relative_limited: dict[str, list[np.ndarray]] = {a: [] for a in arm_names}

    def __len__(self) -> int:
        return len(self.times)

    def record(self, state: RobotState, filtered: Mapping[str, FilterResult]) -> None:
        self.times.append(state.time)
        for arm in self.arm_names:
            f = filtered[arm]
            self.commanded[arm].append(f.target)
            self.measured[arm].append(state.arms[arm].positions)
            self.clipped[arm].append(f.position_clipped)
            self.rate_limited[arm].append(f.rate_limited)
            self.relative_limited[arm].append(f.relative_limited)

    def errors(self, arm: str) -> np.ndarray:
        """(N, 6) commanded - measured [rad]."""
        return np.asarray(self.commanded[arm]) - np.asarray(self.measured[arm])

    def summary(self) -> dict:
        out: dict = {"ticks": len(self)}
        for arm in self.arm_names:
            if not self.times:
                break
            err = self.errors(arm)
            out[arm] = {
                "max_abs_error_rad": dict(zip(JOINT_NAMES, np.abs(err).max(axis=0).round(5).tolist(), strict=True)),
                "rms_error_rad": dict(zip(JOINT_NAMES, np.sqrt((err**2).mean(axis=0)).round(5).tolist(), strict=True)),
                "ticks_position_clipped": int(np.asarray(self.clipped[arm]).any(axis=1).sum()),
                "ticks_rate_limited": int(np.asarray(self.rate_limited[arm]).any(axis=1).sum()),
                "ticks_relative_limited": int(np.asarray(self.relative_limited[arm]).any(axis=1).sum()),
            }
        return out

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        arrays = {"time": np.asarray(self.times), "joint_names": np.asarray(JOINT_NAMES)}
        for arm in self.arm_names:
            arrays[f"{arm}/commanded"] = np.asarray(self.commanded[arm])
            arrays[f"{arm}/measured"] = np.asarray(self.measured[arm])
            arrays[f"{arm}/position_clipped"] = np.asarray(self.clipped[arm])
            arrays[f"{arm}/rate_limited"] = np.asarray(self.rate_limited[arm])
            arrays[f"{arm}/relative_limited"] = np.asarray(self.relative_limited[arm])
        np.savez_compressed(directory / "tracking.npz", **arrays)
        (directory / "tracking_summary.json").write_text(json.dumps(self.summary(), indent=2), encoding="utf-8")
        return directory


class ArmController:
    def __init__(
        self,
        robot: RobotInterface,
        safety: SafetyConfig,
        poses: Mapping[str, tuple[float, ...]] | None = None,
        on_step: StepCallback | None = None,
    ) -> None:
        self.robot = robot
        self.safety = safety
        self.poses = dict(poses or {})
        self.on_step = on_step
        self.filters = {
            arm: SafetyFilter(
                SafetyLimits.from_config(robot.joint_limits(arm), safety),
                robot.control_dt,
                safety.on_out_of_range,
                name=arm,
            )
            for arm in robot.arm_names
        }
        self.log = TrackingLog(robot.arm_names)
        self.last_compute_s = 0.0
        self._warned_tracking = False
        self.sync()

    @classmethod
    def from_config(cls, robot: RobotInterface, config: RobotConfig, on_step: StepCallback | None = None) -> ArmController:
        return cls(robot, config.safety, config.poses, on_step)

    @property
    def arm_names(self) -> tuple[str, ...]:
        return self.robot.arm_names

    @property
    def dt(self) -> float:
        return self.robot.control_dt

    def limits(self, arm: str) -> SafetyLimits:
        return self.filters[arm].limits

    # --- state ----------------------------------------------------------------

    def sync(self) -> None:
        """Re-initialise the safety filters from the robot's current targets."""
        for arm, q in self.robot.get_targets().items():
            self.filters[arm].reset(q)

    def resolve_pose(self, pose: PoseLike) -> dict[str, np.ndarray]:
        """Pose name / single (6,) array (applied to every arm) / per-arm dict -> per-arm dict."""
        if isinstance(pose, str):
            if pose not in self.poses:
                raise CommandError(f"unknown pose {pose!r}; known: {sorted(self.poses)}")
            pose = np.asarray(self.poses[pose], float)
        if isinstance(pose, Mapping):
            unknown = set(pose) - set(self.arm_names)
            if unknown:
                raise CommandError(f"unknown arms {sorted(unknown)}")
            return {arm: np.asarray(q, float) for arm, q in pose.items()}
        return {arm: np.asarray(pose, float) for arm in self.arm_names}

    def reset(self, pose: PoseLike = "rest") -> RobotState:
        """Simulation reset: place the robot at ``pose`` instantly, then sync filters.

        Soft limits are applied so the robot never starts outside them.
        """
        targets = {arm: self.filters[arm].check_range(q)[0] for arm, q in self.resolve_pose(pose).items()}
        state = self.robot.reset(targets)
        self.sync()
        self._warned_tracking = False
        return state

    def get_state(self) -> RobotState:
        return self.robot.get_state()

    # --- one control tick -----------------------------------------------------

    def command(self, targets: Mapping[str, np.ndarray]) -> StepReport:
        """Filter, send, and step one control period. Arms not given hold their target."""
        unknown = set(targets) - set(self.arm_names)
        if unknown:
            raise CommandError(f"unknown arms {sorted(unknown)}")
        t0 = time.perf_counter()
        measured = self.robot.get_state().arms
        filtered = {
            arm: f(targets[arm] if arm in targets else f.last_target, measured[arm].positions)
            for arm, f in self.filters.items()
        }
        self.robot.set_targets({arm: r.target for arm, r in filtered.items()})
        state = self.robot.step()
        self.log.record(state, filtered)
        self._check_tracking(state, filtered)
        # Wall time of filter + send + step, excluding the on_step callback
        # (viewer sync / real-time sleep), which is pacing, not computation.
        self.last_compute_s = time.perf_counter() - t0
        if self.on_step is not None:
            self.on_step(state)
        return StepReport(state, filtered)

    def _check_tracking(self, state: RobotState, filtered: Mapping[str, FilterResult]) -> None:
        if self._warned_tracking:
            return
        for arm, r in filtered.items():
            err = np.abs(r.target - state.arms[arm].positions)
            if err.max() > self.safety.tracking_error_warn:
                j = int(err.argmax())
                log.warning("arm %r: tracking error %.3f rad on %s at t=%.2fs (warn threshold %.3f)",
                            arm, err[j], JOINT_NAMES[j], state.time, self.safety.tracking_error_warn)
                self._warned_tracking = True

    # --- motions --------------------------------------------------------------

    def hold(self, duration: float) -> RobotState:
        state = self.get_state()
        for _ in range(max(1, round(duration / self.dt))):
            state = self.command({}).state
        return state

    def execute_trajectory(self, trajectories: Mapping[str, Trajectory]) -> RobotState:
        """Track per-arm trajectories tick by tick (arms without one hold still)."""
        for arm, traj in trajectories.items():
            current = self.filters[arm].last_target
            if not np.allclose(traj.start, current, atol=0.05):
                raise CommandError(f"arm {arm!r}: trajectory starts {np.abs(traj.start - current).max():.3f} rad "
                                   "away from the current target; use move_to() first")
        duration = max((t.duration for t in trajectories.values()), default=0.0)
        state = self.get_state()
        n = max(1, int(np.ceil(duration / self.dt - 1e-9)))
        for k in range(1, n + 1):
            t = k * self.dt
            state = self.command({arm: traj.sample(t) for arm, traj in trajectories.items()}).state
        return state

    def move_to(self, pose: PoseLike, duration: float | None = None, speed_scale: float = 0.5) -> RobotState:
        """Smooth min-jerk move from the current target to ``pose``.

        If ``duration`` is None it is chosen from the velocity/acceleration
        limits scaled by ``speed_scale``.
        """
        trajs = {}
        for arm, goal in self.resolve_pose(pose).items():
            goal, _ = self.filters[arm].check_range(goal)
            start = self.filters[arm].last_target
            lim = self.limits(arm)
            t = duration if duration is not None else min_jerk_duration(
                start, goal, lim.max_velocity * speed_scale, lim.max_acceleration * speed_scale**2, 0.2)
            trajs[arm] = min_jerk(start, goal, t, self.dt)
        return self.execute_trajectory(trajs)

    def move_through(self, arm: str, waypoints: list[np.ndarray], speed_scale: float = 0.5) -> RobotState:
        lim = self.limits(arm)
        traj = waypoint_trajectory(self.filters[arm].last_target, waypoints, lim.max_velocity,
                                   lim.max_acceleration, self.dt, speed_scale)
        return self.execute_trajectory({arm: traj})

    def move_joint(self, arm: str, joint: str, value: float, duration: float | None = None,
                   speed_scale: float = 0.5) -> RobotState:
        """Move one joint, keeping the others at their current targets."""
        goal = self.filters[arm].last_target
        goal[joint_index(joint)] = value
        return self.move_to({arm: goal}, duration, speed_scale)

    def set_gripper(self, arm: str, opening: float, duration: float | None = None) -> RobotState:
        """Opening in [0, 1]: 0 = closed, 1 = open (over the hard range, as LeRobot's 0..100)."""
        lim = self.robot.joint_limits(arm)
        q = normalized_to_gripper(opening, lim.lower[GRIPPER_INDEX], lim.upper[GRIPPER_INDEX])
        return self.move_joint(arm, "gripper", q, duration)

    def open_gripper(self, arm: str, duration: float | None = None) -> RobotState:
        return self.set_gripper(arm, 1.0, duration)

    def close_gripper(self, arm: str, duration: float | None = None) -> RobotState:
        return self.set_gripper(arm, 0.0, duration)

    def wait_until_settled(self, tol: float = 0.01, timeout: float = 2.0) -> bool:
        """Hold targets until every joint is within ``tol`` rad of its target (or timeout)."""
        for _ in range(max(1, round(timeout / self.dt))):
            report = self.command({})
            if all(np.abs(r.target - report.state.arms[a].positions).max() < tol for a, r in report.filtered.items()):
                return True
        return False
