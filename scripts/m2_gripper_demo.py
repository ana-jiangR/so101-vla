"""Milestone 2: open and close the simulated SO-101 gripper.

Steps through openings 0 (closed) -> 1 (open) and prints the commanded opening,
the measured joint angle, and the jaw gap, which confirms the direction
convention (lower joint limit = closed).

Usage:
    uv run python scripts/m2_gripper_demo.py [--headless]
"""

from __future__ import annotations

import argparse

import numpy as np

from so101_vla.config import load_robot_config
from so101_vla.control import ArmController
from so101_vla.conventions import GRIPPER_INDEX, gripper_to_normalized
from so101_vla.robot.mujoco_robot import MuJoCoRobot
from so101_vla.sim.viewer import RealtimeViewer

ARM = "main"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="configs/robot/so101_single.yaml")
    p.add_argument("--headless", action="store_true")
    args = p.parse_args()

    cfg = load_robot_config(args.config)
    robot = MuJoCoRobot.from_config(cfg)
    jaw = robot.model.body(f"{ARM}/moving_jaw_so101_v1").id
    site = robot.model.site(f"{ARM}/gripperframe").id

    with RealtimeViewer(robot, headless=args.headless) as viewer:
        ctrl = ArmController.from_config(robot, cfg, on_step=viewer)
        ctrl.reset("home")
        viewer.restart_clock()
        lim = ctrl.limits(ARM)
        lo, hi = lim.lower[GRIPPER_INDEX], lim.upper[GRIPPER_INDEX]

        print(f"gripper soft range [{lo:+.3f}, {hi:+.3f}] rad  (0 = closed, 1 = open)")
        print(f"{'opening':>8} {'joint [rad]':>12} {'measured opening':>17} {'jaw gap [mm]':>13}")
        for opening in (0.0, 0.25, 0.5, 0.75, 1.0, 0.0, 1.0):
            ctrl.set_gripper(ARM, opening)
            ctrl.wait_until_settled(tol=0.005)
            q = ctrl.get_state().arms[ARM].positions[GRIPPER_INDEX]
            gap = np.linalg.norm(robot.data.xipos[jaw] - robot.data.site_xpos[site]) * 1000
            print(f"{opening:>8.2f} {q:>12.4f} {gripper_to_normalized(q, lo, hi):>17.3f} {gap:>13.1f}")
            ctrl.hold(0.3)

        if not args.headless:
            print("Close the viewer window to exit.")
            viewer.idle()


if __name__ == "__main__":
    main()
