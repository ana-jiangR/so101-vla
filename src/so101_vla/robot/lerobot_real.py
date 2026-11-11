"""Real SO-101 follower behind ``RobotInterface``, using LeRobot's ``SO101Follower`` driver.

This is the "RealRobot" backend planned in design.md (Task 1.5): the same
``ArmController`` / ``SafetyFilter`` / ``CommandSelector`` / ``PolicySource`` stack that
drives the simulation drives the real arm through it.

Units: LeRobot reads and writes ``so101_follower`` degrees (``use_degrees=True``) and the
gripper in 0..100. Internally (like ``MuJoCoRobot``) we use radians; the gripper's 0..100
maps onto the simulated gripper range, so ``conventions.LeRobotUnits`` converts both
backends identically and a policy sees the same numbers.

Joint limits come from the arm's own calibration (``range_min``/``range_max``): in LeRobot's
DEGREES mode a joint spans ±(range_max - range_min)/2 × 360/4095 around 0. The real ranges are
a few degrees wider than the simulation model's.

Safety on connect: LeRobot's ``SO101Follower.connect`` enables torque without setting a goal
first, so a stale ``Goal_Position`` could make the arm jump. ``connect_safely`` does the same
steps as LeRobot's ``connect`` (bus, calibration check, cameras, ``configure``) but writes
``Goal_Position = Present_Position`` before torque comes on.
"""

from __future__ import annotations

import time
from collections.abc import Mapping

import cv2
import numpy as np

from so101_vla.conventions import JOINT_NAMES, LeRobotUnits
from so101_vla.robot.interface import (
    ArmState,
    CommandError,
    JointLimits,
    RobotInterface,
    RobotState,
    check_target_array,
)

# Simulated gripper range [rad] that LeRobot's 0..100 maps onto (MJCF ctrlrange).
SIM_GRIPPER_RANGE = (-0.174533, 1.745329)
STS3215_RESOLUTION = 4095


def calibration_limits(calibration: Mapping) -> JointLimits:
    """Hard joint limits [rad] from a LeRobot calibration (DEGREES mode, gripper on the sim range)."""
    lower, upper = np.zeros(6), np.zeros(6)
    for i, name in enumerate(JOINT_NAMES):
        if name == "gripper":
            lower[i], upper[i] = SIM_GRIPPER_RANGE
            continue
        c = calibration[name]
        half = np.deg2rad((c.range_max - c.range_min) / 2 * 360 / STS3215_RESOLUTION)
        lower[i], upper[i] = -half, half
    return JointLimits(lower=lower, upper=upper)


def square_resize(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Center-crop to the target aspect ratio, then resize (RGB uint8)."""
    h, w = img.shape[:2]
    target = width / height
    if w / h > target:
        cw = int(round(h * target))
        x = (w - cw) // 2
        img = img[:, x:x + cw]
    else:
        ch = int(round(w / target))
        y = (h - ch) // 2
        img = img[y:y + ch]
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


class LeRobotSO101(RobotInterface):
    """One real SO-101 follower arm (``arm`` name ``"main"``) plus its LeRobot cameras."""

    def __init__(self, port: str, robot_id: str, cameras: Mapping | None = None, control_hz: float = 30.0,
                 max_relative_target: float | None = 5.0, arm: str = "main") -> None:
        """
        cameras:              {name: lerobot OpenCVCameraConfig}
        max_relative_target:  LeRobot follower option [deg]: each sent goal is clamped to within this
                              of the measured position (a second, driver-level safeguard)
        """
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        self._robot = SO101Follower(SO101FollowerConfig(
            port=port, id=robot_id, cameras=dict(cameras or {}), max_relative_target=max_relative_target))
        if not self._robot.calibration:
            raise CommandError(f"no LeRobot calibration for id {robot_id!r}; run lerobot-calibrate first")
        self._arm = arm
        self._dt = 1.0 / control_hz
        self._limits = calibration_limits(self._robot.calibration)
        self._units = LeRobotUnits(self._limits.lower, self._limits.upper)
        self._targets: np.ndarray | None = None
        self._prev: tuple[float, np.ndarray] | None = None
        self._t0 = time.monotonic()
        self._last_step = time.monotonic()
        self._connected = False

    # --- connection ---------------------------------------------------------------

    def connect_safely(self) -> None:
        r = self._robot
        r.bus.connect()
        if not r.is_calibrated:
            # Same as LeRobot's connect(): write the saved calibration to the motors (no prompt).
            r.bus.write_calibration(r.calibration)
        present = r.bus.sync_read("Present_Position", normalize=False)
        r.bus.sync_write("Goal_Position", present, normalize=False)  # hold here when torque comes on
        for cam in r.cameras.values():
            cam.connect()
        r.configure()  # LeRobot: position mode, PID, gripper current limits; re-enables torque
        self._connected = True
        q = self.get_state().arms[self._arm].positions
        self._targets = q.copy()
        self._last_step = time.monotonic()

    def close(self, release_torque: bool = True) -> None:
        if not self._connected:
            return
        for cam in self._robot.cameras.values():
            try:
                cam.disconnect()
            except Exception:  # noqa: BLE001
                pass
        self._robot.bus.disconnect(disable_torque=release_torque)
        self._connected = False

    def hold_here(self) -> None:
        """Freeze: goal = measured position (e.g. after an interrupt)."""
        q = self.get_state().arms[self._arm].positions
        self._send(q)

    # --- RobotInterface -------------------------------------------------------------

    @property
    def arm_names(self) -> tuple[str, ...]:
        return (self._arm,)

    @property
    def control_dt(self) -> float:
        return self._dt

    def joint_limits(self, arm: str) -> JointLimits:
        self._check_arm(arm)
        return self._limits

    def get_state(self) -> RobotState:
        pos = self._robot.bus.sync_read("Present_Position")
        q = self._units.from_lerobot(np.array([pos[j] for j in JOINT_NAMES], dtype=np.float64))
        t = time.monotonic() - self._t0
        v = np.zeros(6) if self._prev is None else (q - self._prev[1]) / max(t - self._prev[0], 1e-3)
        self._prev = (t, q)
        return RobotState(time=t, arms={self._arm: ArmState(positions=q, velocities=v)})

    def get_targets(self) -> dict[str, np.ndarray]:
        if self._targets is None:
            return {self._arm: self.get_state().arms[self._arm].positions}
        return {self._arm: self._targets.copy()}

    def set_targets(self, targets: Mapping[str, np.ndarray]) -> None:
        for arm, target in targets.items():
            self._check_arm(arm)
            q = check_target_array(target, arm)
            inside = self._limits.contains(q)
            if not inside.all():
                bad = [JOINT_NAMES[i] for i in np.flatnonzero(~inside)]
                raise CommandError(f"arm {arm!r}: target outside hard limits: {bad}")
            self._send(q)

    def _send(self, q: np.ndarray) -> None:
        lr = self._units.to_lerobot(q)
        self._robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINT_NAMES, lr, strict=True)})
        self._targets = q.copy()

    def step(self) -> RobotState:
        """Wait out the rest of the control period (real time), then read the state."""
        remaining = self._dt - (time.monotonic() - self._last_step)
        if remaining > 0:
            time.sleep(remaining)
        self._last_step = time.monotonic()
        return self.get_state()

    def reset(self, pose: Mapping[str, np.ndarray]) -> RobotState:
        raise CommandError("a real arm cannot be teleported; use ArmController.move_to()")

    # --- cameras ----------------------------------------------------------------------

    @property
    def camera_names(self) -> tuple[str, ...]:
        return tuple(self._robot.cameras)

    def render_camera(self, name: str, width: int = 640, height: int = 480) -> np.ndarray:
        if name not in self._robot.cameras:
            raise CommandError(f"unknown camera {name!r}; robot has {self.camera_names}")
        return square_resize(np.asarray(self._robot.cameras[name].read_latest()), width, height)

    def _check_arm(self, arm: str) -> None:
        if arm != self._arm:
            raise CommandError(f"unknown arm {arm!r}; robot has ({self._arm!r},)")
