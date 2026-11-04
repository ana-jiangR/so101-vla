"""Milestone 4: the tabletop task scene and its cameras (Tasks 2.1, 2.2).

Resets the one-cube pick-and-place scene (LeRobot's official tabletop, see
assets/lerobot_tabletop/PROVENANCE.md) with a few seeds, renders every camera,
and saves the images with the seeded layout and the camera timestamps.

    uv run python scripts/m4_tabletop_scene.py                 # images + viewer
    uv run python scripts/m4_tabletop_scene.py --headless --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from so101_vla.config import PROJECT_ROOT, load_robot_config
from so101_vla.control import ArmController
from so101_vla.robot.mujoco_robot import MuJoCoRobot
from so101_vla.sim.tabletop import PickPlaceTask
from so101_vla.sim.viewer import RealtimeViewer, ViewerClosed
from so101_vla.utils.images import save_png, tile


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/robot/so101_tabletop.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    cfg = load_robot_config(args.config)
    robot = MuJoCoRobot.from_config(cfg)
    ctrl = ArmController.from_config(robot, cfg)
    task = PickPlaceTask(robot)
    out = PROJECT_ROOT / "outputs" / f"tabletop_scene_{time.strftime('%Y%m%d_%H%M%S')}"
    print(f"cameras: {robot.camera_names}; instruction in scene: {task.scene_instruction!r}")

    meta = []
    for seed in args.seeds:
        s = task.reset(np.asarray(cfg.poses["home"]), seed=seed)
        ctrl.sync()
        images = {cam: robot.render_camera(cam, args.width, args.height) for cam in robot.camera_names}
        for cam, img in images.items():
            save_png(out / f"seed{seed}_{cam.replace('/', '_')}.png", img)
        save_png(out / f"seed{seed}_all.png", tile(list(images.values()), 2))
        meta.append({"seed": seed, "camera_timestamp_s": s.time, "layout": task.layout,
                     "gripper_to_cube_m": round(s.gripper_to_cube, 4), "cube_in_box": s.in_box})
        print(f"seed {seed}: cube {task.layout['cube_xy']} yaw {task.layout['cube_yaw']:+.2f}, "
              f"box {task.layout['box_xy']}, gripper-cube {s.gripper_to_cube * 100:.1f} cm")
    (out / "scene.json").write_text(json.dumps({"cameras": list(robot.camera_names), "width": args.width,
                                                "height": args.height, "resets": meta}, indent=2), encoding="utf-8")
    print(f"images saved to {out.relative_to(PROJECT_ROOT)}")

    if not args.headless:
        print("viewer open: arm holds home; close the window to exit")
        try:
            with RealtimeViewer(robot) as viewer:
                ctrl.on_step = viewer
                viewer.restart_clock()
                while True:
                    ctrl.hold(1.0)
        except (ViewerClosed, KeyboardInterrupt):
            pass
    robot.close()


if __name__ == "__main__":
    main()
