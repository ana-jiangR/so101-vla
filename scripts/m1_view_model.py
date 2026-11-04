"""Milestone 1: load the SO-101 scene and open the MuJoCo viewer.

The arm holds the chosen pose under its position actuators (gravity on), so
you can check that the model loads, looks right, and stands still. In the
viewer, double-click a body and Ctrl+right-drag to push it.

Usage:
    uv run python scripts/m1_view_model.py [--pose rest|home|zero] [--config PATH]
"""

from __future__ import annotations

import argparse

from so101_vla.config import load_robot_config
from so101_vla.control import ArmController
from so101_vla.robot.mujoco_robot import MuJoCoRobot
from so101_vla.sim.viewer import RealtimeViewer


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="configs/robot/so101_single.yaml")
    p.add_argument("--pose", default="rest")
    args = p.parse_args()

    cfg = load_robot_config(args.config)
    robot = MuJoCoRobot.from_config(cfg)
    m = robot.model
    print(f"Loaded {cfg.scene.name} with arms {robot.arm_names}: nq={m.nq} nu={m.nu} "
          f"physics {1 / m.opt.timestep:.0f} Hz, control {1 / robot.control_dt:.0f} Hz")

    controller = ArmController.from_config(robot, cfg)
    controller.reset(args.pose)
    print(f"Holding pose {args.pose!r}. Close the viewer window to exit.")
    with RealtimeViewer(robot) as viewer:
        viewer.idle()


if __name__ == "__main__":
    main()
