"""Backend-independent robot interface.

A robot is a set of named arms (e.g. ``"main"``, or ``"left"``/``"right"``),
each with the six joints in ``conventions.JOINT_NAMES`` order, in radians.
Every method that takes or returns per-arm values uses a dict keyed by arm
name, so one- and two-arm robots share the same API.

Backends only *execute* position targets and report state. Trajectory
generation and soft safety limits live in ``so101_vla.control`` and work with
any backend. Backends must still reject malformed or out-of-hard-limit targets
(defense in depth).

Implementations:
    MuJoCoRobot     so101_vla.robot.mujoco_robot
    (real robot)    future, e.g. wrapping LeRobot's SO-101 follower driver
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from so101_vla.conventions import JOINT_NAMES, NUM_JOINTS, LeRobotUnits


class CommandError(ValueError):
    """A motion command was malformed or violated a limit."""


@dataclass(frozen=True)
class JointLimits:
    """Hard position limits of one arm [rad], in canonical joint order."""

    lower: np.ndarray
    upper: np.ndarray

    def contains(self, q: np.ndarray, tol: float = 1e-9) -> np.ndarray:
        return (q >= self.lower - tol) & (q <= self.upper + tol)


@dataclass(frozen=True)
class ArmState:
    positions: np.ndarray  # (6,) rad
    velocities: np.ndarray  # (6,) rad/s


@dataclass(frozen=True)
class RobotState:
    time: float  # seconds since reset (sim time for MuJoCo)
    arms: dict[str, ArmState]


def check_target_array(target: np.ndarray, arm: str) -> np.ndarray:
    """Validate shape/dtype/finiteness of one arm's joint target."""
    q = np.asarray(target, dtype=np.float64)
    if q.shape != (NUM_JOINTS,):
        raise CommandError(f"arm {arm!r}: target must have shape ({NUM_JOINTS},) {JOINT_NAMES}, got {q.shape}")
    if not np.all(np.isfinite(q)):
        raise CommandError(f"arm {arm!r}: target contains NaN/inf: {q}")
    return q


class RobotInterface(ABC):
    """Position-controlled robot made of one or more SO-101 arms."""

    joint_names: tuple[str, ...] = JOINT_NAMES

    @property
    @abstractmethod
    def arm_names(self) -> tuple[str, ...]: ...

    @property
    @abstractmethod
    def control_dt(self) -> float:
        """Duration of one ``step()`` [s]."""

    @abstractmethod
    def joint_limits(self, arm: str) -> JointLimits: ...

    @abstractmethod
    def get_state(self) -> RobotState: ...

    @abstractmethod
    def get_targets(self) -> dict[str, np.ndarray]:
        """Most recently commanded position targets."""

    @abstractmethod
    def set_targets(self, targets: Mapping[str, np.ndarray]) -> None:
        """Set position targets for some or all arms (applied from the next step).

        Raises CommandError for unknown arms, bad shapes, NaN/inf, or values
        outside the hard limits. Arms not in ``targets`` keep their previous target.
        """

    @abstractmethod
    def step(self) -> RobotState:
        """Advance one control period and return the new state."""

    @abstractmethod
    def reset(self, pose: Mapping[str, np.ndarray]) -> RobotState:
        """Return to a known state with the given pose as position target.

        Simulation backends may place the robot there instantly. A hardware
        backend must not teleport: use ``ArmController.move_to`` instead.
        """

    # --- cameras / observations (optional for backends without cameras) ------

    @property
    def camera_names(self) -> tuple[str, ...]:
        """Cameras ``render_camera`` accepts (empty if the backend has none)."""
        return ()

    def render_camera(self, name: str, width: int = 640, height: int = 480) -> np.ndarray:
        """Current image from camera ``name`` as an (height, width, 3) uint8 RGB array."""
        raise CommandError(f"unknown camera {name!r}; robot has {self.camera_names}")

    def get_observation(self, cameras: Sequence[str] = (), width: int = 640, height: int = 480,
                        norm: str = "degrees") -> dict:
        """Observation in LeRobot's format, as ``SO101Follower.get_observation()`` returns it.

        Joints are ``"<joint>.pos"`` in LeRobot units (``norm``: "degrees" or
        "range_m100_100" for the arm joints, gripper 0..100); with several arms the
        keys get an ``"<arm>_"`` prefix, as LeRobot's ``bi_so_follower`` does. Each
        requested camera adds ``{name: (H, W, 3) uint8}``. Rendering is skipped
        unless cameras are asked for, so state-only callers stay cheap.
        """
        state = self.get_state()
        obs: dict = {}
        for arm, arm_state in state.arms.items():
            lim = self.joint_limits(arm)
            prefix = "" if len(self.arm_names) == 1 else f"{arm}_"
            obs.update(LeRobotUnits(lim.lower, lim.upper, norm).to_dict(arm_state.positions, prefix))
        for cam in cameras:
            obs[cam] = self.render_camera(cam, width, height)
        return obs

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release resources (viewer, serial ports, ...)."""

    def __enter__(self) -> RobotInterface:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
