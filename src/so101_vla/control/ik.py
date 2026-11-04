"""Inverse kinematics for one SO-101 arm with mink (Stage 3 scripted demonstrations).

Uses mink's documented differential-IK loop (github.com/kevinzakka/mink): a
``FrameTask`` on the Menagerie ``gripperframe`` site plus a low-weight
``PostureTask``, ``ConfigurationLimit`` for the joint limits, ``solve_ik`` with
the DAQP solver, ``Configuration.integrate_inplace``. LeRobot's own
``RobotKinematics`` (placo + URDF) would be the first choice, but placo has no
Windows wheels.

IK runs on a separate, arm-only copy of the Menagerie MJCF, so it never touches
the simulation. Targets are given in the world frame and mapped into the arm's
base frame from the live simulation. The result is only a joint target: it still
goes through the command selector, the safety filter and the actuators.

Frame convention (``gripperframe``, checked at HOME): site x points out of the
jaws (approach direction); site y is the jaw closing direction.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from so101_vla.conventions import ARM_JOINT_NAMES, EE_SITE
from so101_vla.robot.mujoco_robot import MuJoCoRobot
from so101_vla.sim.scene import arm_prefix


def top_down_rotation(yaw: float) -> np.ndarray:
    """Site orientation for a vertical grasp: approach (x) straight down, jaws closing along ``yaw``."""
    x = np.array([0.0, 0.0, -1.0])
    y = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    return np.column_stack([x, y, np.cross(x, y)])


@dataclass(frozen=True)
class IKResult:
    q: np.ndarray  # 5 arm joints [rad]
    position_error: float  # [m]
    orientation_error: float  # [rad]


class ArmIK:
    def __init__(self, robot: MuJoCoRobot, arm: str, arm_model_path: str, position_cost: float = 1.0,
                 orientation_cost: float = 0.2, posture_cost: float = 1e-3,
                 limit_margin: float = 0.02) -> None:
        """``limit_margin`` [rad] keeps solutions inside the soft limits (robot config ``position_margin``)."""
        import mink

        self._mink = mink
        self.model = mujoco.MjModel.from_xml_path(str(arm_model_path))
        if self.model.nq != 6:
            raise ValueError("expected the arm-only SO-101 model (6 joints)")
        self.configuration = mink.Configuration(self.model)
        self.ee_task = mink.FrameTask(EE_SITE, "site", position_cost=position_cost,
                                      orientation_cost=orientation_cost, lm_damping=1.0)
        self.posture_task = mink.PostureTask(self.model, cost=posture_cost)
        self.limits = [mink.ConfigurationLimit(self.model, min_distance_from_limits=limit_margin)]
        self.orientation_cost = orientation_cost
        # World -> arm base frame, from the live simulation (the arm's mount).
        base = robot.model.body(arm_prefix(arm) + "base").id
        R = robot.data.xmat[base].reshape(3, 3).copy()
        p = robot.data.xpos[base].copy()
        self._to_base = (R.T, -R.T @ p)
        self._qadr = np.array([self.model.joint(j).qposadr[0] for j in (*ARM_JOINT_NAMES, "gripper")])

    def solve(self, q_init: np.ndarray, pos_world: np.ndarray, rot_world: np.ndarray | None,
              iters: int = 400, dt: float = 0.05) -> IKResult:
        """Joint angles reaching ``pos_world`` [m] with ``rot_world`` (None = position only), from ``q_init``."""
        mink = self._mink
        Rb, pb = self._to_base
        pos = Rb @ np.asarray(pos_world, float) + pb
        q = np.zeros(self.model.nq)
        q[self._qadr] = np.asarray(q_init, float)[:6]
        self.configuration.update(q)
        self.posture_task.set_target(q)
        self.ee_task.set_orientation_cost(0.0 if rot_world is None else self.orientation_cost)
        rot = np.eye(3) if rot_world is None else Rb @ np.asarray(rot_world, float)
        self.ee_task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(rot), pos))
        err = np.full(6, np.inf)
        for _ in range(iters):
            vel = mink.solve_ik(self.configuration, [self.ee_task, self.posture_task], dt, "daqp",
                                damping=1e-3, limits=self.limits)
            self.configuration.integrate_inplace(vel, dt)
            err = self.ee_task.compute_error(self.configuration)
            if np.linalg.norm(err[:3]) < 1e-4 and (rot_world is None or np.linalg.norm(err[3:]) < 1e-3):
                break
        qa = self.configuration.q[self._qadr][:5].copy()
        return IKResult(qa, float(np.linalg.norm(err[:3])), 0.0 if rot_world is None else float(np.linalg.norm(err[3:])))
