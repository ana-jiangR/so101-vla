"""Optional interactive viewer that follows a MuJoCoRobot in real time."""

from __future__ import annotations

import time

import mujoco
import mujoco.viewer

from so101_vla.robot.interface import RobotState
from so101_vla.robot.mujoco_robot import MuJoCoRobot


class ViewerClosed(Exception):
    """Raised from the step callback when the user closes the viewer window."""


class RealtimeViewer:
    """Passive MuJoCo viewer used as an ``ArmController`` step callback.

    Each control tick it syncs the window and sleeps so that sim time advances
    at ``speed`` x wall-clock time. Use ``headless=True`` to make it a no-op.

        with RealtimeViewer(robot) as viewer:
            controller = ArmController.from_config(robot, cfg, on_step=viewer)
    """

    def __init__(self, robot: MuJoCoRobot, headless: bool = False, speed: float = 1.0) -> None:
        self.robot = robot
        self.headless = headless
        self.speed = speed
        self._handle = None
        self._t0_wall = 0.0
        self._t0_sim = 0.0

    def __enter__(self) -> RealtimeViewer:
        if not self.headless:
            self._handle = mujoco.viewer.launch_passive(self.robot.model, self.robot.data)
            self._handle.cam.lookat[:] = [0.15, 0.0, 0.12]
            self._handle.cam.distance = 0.8
            self._handle.cam.azimuth = 135
            self._handle.cam.elevation = -20
        self.restart_clock()
        return self

    def __exit__(self, *exc: object) -> bool:
        if self._handle is not None:
            self._handle.close()
        return exc[0] is ViewerClosed  # closing the window ends the demo quietly

    @property
    def is_running(self) -> bool:
        return self.headless or (self._handle is not None and self._handle.is_running())

    def restart_clock(self) -> None:
        self._t0_wall = time.perf_counter()
        self._t0_sim = self.robot.data.time

    def __call__(self, state: RobotState) -> None:
        if self.headless:
            return
        if not self._handle.is_running():
            raise ViewerClosed
        self._handle.sync()
        # Pace: sim time elapsed / speed should equal wall time elapsed.
        ahead = (state.time - self._t0_sim) / self.speed - (time.perf_counter() - self._t0_wall)
        if ahead > 0:
            time.sleep(ahead)
        elif ahead < -0.25:  # fell behind (e.g. window dragged); don't try to catch up
            self.restart_clock()

    def idle(self) -> None:
        """Keep the window responsive while the robot holds still, until closed."""
        while self.is_running and not self.headless:
            mujoco.mj_step(self.robot.model, self.robot.data, nstep=self.robot.n_substeps)
            self(self.robot.get_state())
