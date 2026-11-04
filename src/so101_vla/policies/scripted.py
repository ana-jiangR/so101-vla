"""Scripted pick-and-place expert for recording demonstrations (Stage 3, design.md step 1).

design.md Stage 3 asks for "simulated teleoperation or an expert control policy". This
expert uses privileged simulator state (the cube and box poses after reset), plans
grasp/place poses with IK (``control.ik``), and turns them into one smooth joint
trajectory (Stage 1 min-jerk segments). It is then played back as an ordinary command
source (``TrajectoryLeader``), so it drives the follower exactly like a leader arm:
``CommandSelector`` -> safety filter -> actuators. The recorded "action" is what a
teleoperator would have sent.

Plan: pregrasp above the cube -> descend -> close -> lift -> above the box -> lower ->
open -> retreat -> home. The jaws close across a pair of cube faces chosen to be roughly
perpendicular to the arm's reach direction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from so101_vla.control.ik import ArmIK, top_down_rotation
from so101_vla.control.trajectory import Trajectory, min_jerk, min_jerk_duration
from so101_vla.conventions import LeRobotUnits
from so101_vla.sim.tabletop import PickPlaceTask
from so101_vla.teleop.leaders import TrajectoryLeader


@dataclass(frozen=True)
class ExpertConfig:
    pregrasp_height: float = 0.08  # gripperframe z [m] above the table
    grasp_height: float = 0.012
    carry_height: float = 0.09  # clears the 3.5 cm box walls with the cube in hand
    release_height: float = 0.06
    gripper_open: float = 0.55  # opening 0..1 (0 = closed)
    gripper_closed: float = 0.0
    hold_s: float = 0.3  # pause after each waypoint (lets the gripper close/open)
    speed_scale: tuple[float, ...] = (0.5, 0.3, 0.3, 0.3, 0.4, 0.3, 0.4, 0.4, 0.5)  # per segment
    max_ik_error: float = 0.015  # [m]; worse than this -> plan rejected


class PlanningError(RuntimeError):
    pass


def _grasp_yaw(cube_xy: np.ndarray, cube_yaw: float) -> float:
    """Cube face normal (yaw + k*90 deg) closest to perpendicular to the arm's reach direction."""
    radial = np.arctan2(cube_xy[1], cube_xy[0]) + np.pi / 2
    candidates = cube_yaw + np.arange(-2, 3) * np.pi / 2
    return float(candidates[np.argmin(np.abs(np.angle(np.exp(1j * (candidates - radial)))))])


def plan_pick_place(task: PickPlaceTask, ik: ArmIK, q_start: np.ndarray, home: np.ndarray,
                    max_velocity: np.ndarray, max_acceleration: np.ndarray, dt: float,
                    cfg: ExpertConfig | None = None,
                    position_margin: np.ndarray | None = None) -> tuple[Trajectory, list[dict]]:
    """Joint trajectory [rad] for the current task layout, plus a per-waypoint report.

    Waypoints stay inside the soft limits (hard limits shrunk by ``position_margin``),
    so the safety filter never has to clip the expert.
    """
    cfg = cfg or ExpertConfig()
    s = task.state()
    cube = s.cube_pos
    box = np.array(task.layout["box_xy"])
    gyaw = _grasp_yaw(cube[:2], task.layout["cube_yaw"])
    byaw = float(np.arctan2(box[1], box[0]) + np.pi / 2)
    lim = task.robot.joint_limits(task.arm)
    margin = np.zeros(6) if position_margin is None else np.asarray(position_margin, float)
    lo, hi = lim.lower + margin, lim.upper - margin

    def g(opening: float) -> float:
        return float(lim.lower[5] + opening * (lim.upper[5] - lim.lower[5]))

    steps = [  # (name, target position or None = keep pose, yaw, gripper opening)
        ("pregrasp", [*cube[:2], cfg.pregrasp_height], gyaw, cfg.gripper_open),
        ("descend", [*cube[:2], cfg.grasp_height], gyaw, cfg.gripper_open),
        ("close", None, None, cfg.gripper_closed),
        ("lift", [*cube[:2], cfg.carry_height], gyaw, cfg.gripper_closed),
        ("over_box", [*box, cfg.carry_height], byaw, cfg.gripper_closed),
        ("lower", [*box, cfg.release_height], byaw, cfg.gripper_closed),
        ("open", None, None, cfg.gripper_open),
        ("retreat", [*box, cfg.carry_height + 0.01], byaw, cfg.gripper_open),
        ("home", "home", None, None),
    ]
    q = np.asarray(q_start, float).copy()
    traj = Trajectory(np.array([0.0]), q[None, :])
    report = []
    for (name, pos, yaw, opening), speed in zip(steps, cfg.speed_scale, strict=True):
        if isinstance(pos, str):
            q = np.asarray(home, float).copy()
            err = 0.0
        else:
            if pos is not None:
                res = ik.solve(q, np.asarray(pos, float), top_down_rotation(yaw))
                if res.position_error > cfg.max_ik_error:
                    raise PlanningError(f"{name}: IK position error {res.position_error * 1000:.1f} mm")
                q = np.append(res.q, q[5])
                err = res.position_error
            else:
                err = 0.0
            q[5] = g(opening)
        q = np.clip(q, lo, hi)
        vmax, amax = np.asarray(max_velocity) * speed, np.asarray(max_acceleration) * speed**2
        traj = traj.then(min_jerk(traj.end, q, min_jerk_duration(traj.end, q, vmax, amax, 0.2), dt))
        n_hold = max(1, int(round(cfg.hold_s / dt)))
        traj = traj.then(Trajectory(np.arange(n_hold + 1) * dt, np.tile(traj.end, (n_hold + 1, 1))))
        report.append({"step": name, "ik_error_mm": round(err * 1000, 2), "t_end": round(float(traj.times[-1]), 3)})
    return traj, report


def expert_leader(task: PickPlaceTask, ik: ArmIK, q_start: np.ndarray, home: np.ndarray, safety,
                  dt: float, clock, cfg: ExpertConfig | None = None) -> tuple[TrajectoryLeader, list[dict]]:
    """The planned trajectory as a LeRobot-style command source (LeRobot units)."""
    traj, report = plan_pick_place(task, ik, q_start, home, safety.max_velocity, safety.max_acceleration, dt, cfg,
                                   position_margin=safety.position_margin)
    lim = task.robot.joint_limits(task.arm)
    values = LeRobotUnits(lim.lower, lim.upper).to_lerobot(traj.positions)
    leader = TrajectoryLeader(traj.times - traj.times[0], values, clock=clock, id="scripted_expert")
    return leader, report
