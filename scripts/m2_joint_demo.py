"""Milestone 2: move the simulated SO-101 joint by joint, then along a trajectory.

Sequence: rest -> home -> each joint +/-0.4 rad in turn -> gripper close/open
-> coordinated waypoint trajectory -> rest. Every command goes through
ArmController (safety filter + logging). A tracking-error log is saved to
outputs/joint_demo_<timestamp>/.

Usage:
    uv run python scripts/m2_joint_demo.py            # with viewer, real time
    uv run python scripts/m2_joint_demo.py --headless # no window, as fast as possible
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime

import numpy as np

from so101_vla.config import PROJECT_ROOT, load_robot_config
from so101_vla.control import ArmController
from so101_vla.conventions import JOINT_NAMES
from so101_vla.robot.mujoco_robot import MuJoCoRobot
from so101_vla.sim.viewer import RealtimeViewer, ViewerClosed

ARM = "main"
SWEEP = 0.4  # rad; from "home" this keeps every single-joint move clear of the floor

# Coordinated motion through the workspace (rad, canonical joint order).
WAYPOINTS = [
    [0.6, -0.3, 0.3, 1.0, 0.8, 1.2],
    [0.6, -0.1, 0.6, 0.6, -0.6, 0.2],
    [-0.6, -0.1, 0.6, 0.6, 0.6, 1.2],
    [-0.6, -0.5, 0.4, 1.1, -0.4, 0.2],
    [0.0, -1.047198, 1.047198, 0.610865, 0.0, 0.401426],  # back to "home"
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="configs/robot/so101_single.yaml")
    p.add_argument("--headless", action="store_true", help="no viewer window")
    p.add_argument("--speed", type=float, default=0.5, help="fraction of the velocity limits to use (0, 1]")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_robot_config(args.config)
    robot = MuJoCoRobot.from_config(cfg)

    with RealtimeViewer(robot, headless=args.headless) as viewer:
        ctrl = ArmController.from_config(robot, cfg, on_step=viewer)
        ctrl.reset("rest")
        viewer.restart_clock()
        settled = False
        stopped_early = False
        try:
            ctrl.hold(0.5)

            print("-> home")
            ctrl.move_to("home", speed_scale=args.speed)
            home = np.asarray(cfg.poses["home"])

            for j, joint in enumerate(JOINT_NAMES[:-1]):
                print(f"-> {joint}: {home[j] + SWEEP:+.2f}, {home[j] - SWEEP:+.2f}, back")
                for value in (home[j] + SWEEP, home[j] - SWEEP, home[j]):
                    ctrl.move_joint(ARM, joint, value, speed_scale=args.speed)
                ctrl.hold(0.2)

            print("-> gripper: close, open, half")
            ctrl.close_gripper(ARM)
            ctrl.open_gripper(ARM)
            ctrl.set_gripper(ARM, 0.5)
            ctrl.hold(0.2)

            print("-> waypoint trajectory")
            ctrl.move_through(ARM, [np.asarray(w) for w in WAYPOINTS], speed_scale=args.speed)

            print("-> rest")
            ctrl.move_to("rest", speed_scale=args.speed)
            settled = ctrl.wait_until_settled()
        except (ViewerClosed, KeyboardInterrupt):
            stopped_early = True
            print("\nStopped early; saving logs of what ran so far.")

        out = PROJECT_ROOT / "outputs" / f"joint_demo_{datetime.now():%Y%m%d_%H%M%S}"
        ctrl.log.save(out)
        if not ctrl.log.times:
            print(f"nothing ran; empty log saved to {out.relative_to(PROJECT_ROOT)}")
            return
        summary = ctrl.log.summary()[ARM]
        print(f"\nDone: {ctrl.log.summary()['ticks']} control ticks, {robot.data.time:.1f} s sim time, "
              f"settled at rest: {settled}")
        print("max |commanded - measured| per joint [rad]:")
        print(json.dumps(summary["max_abs_error_rad"], indent=2))
        print(f"ticks clipped by position limits: {summary['ticks_position_clipped']}")
        print(f"log saved to {out.relative_to(PROJECT_ROOT)}")
        if not args.headless and not stopped_early:
            print("Close the viewer window to exit.")
            viewer.idle()


if __name__ == "__main__":
    main()
