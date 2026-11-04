"""Milestone 8 (Stage 3): record scripted pick-and-place demonstrations as a LeRobotDataset.

Each episode: seeded reset of the tabletop task -> the scripted expert plans an IK
pick-and-place -> it drives the follower through CommandSelector + safety filter ->
every 30 Hz tick is recorded (state, top/wrist/front images, action, task). Only
episodes where the cube ends in the box (LeRobot env definition) are kept.

    uv sync --extra train
    uv run python scripts/m8_record_demos.py --episodes 10            # pilot dataset
    uv run python scripts/m8_record_demos.py --episodes 50 --root outputs/datasets/so101_pick_place_sim_50
    uv run python scripts/m8_record_demos.py --episodes 1 --viewer    # watch one
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from so101_vla.config import PROJECT_ROOT, load_robot_config, resolve_path
from so101_vla.control import ArmController
from so101_vla.control.ik import ArmIK
from so101_vla.policies.scripted import PlanningError, expert_leader
from so101_vla.recording import DemoRecorder, RecordingConfig
from so101_vla.robot.mujoco_robot import MuJoCoRobot
from so101_vla.sim.tabletop import PickPlaceTask
from so101_vla.sim.viewer import RealtimeViewer, ViewerClosed
from so101_vla.teleop import CommandSelector

TASK = "Pick up the red cube and place it in the green box."


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/robot/so101_tabletop.yaml")
    ap.add_argument("--episodes", type=int, default=10, help="successful episodes to record")
    ap.add_argument("--start-seed", type=int, default=1000, help="seeds start here (eval uses 0..99)")
    ap.add_argument("--root", default="outputs/datasets/so101_pick_place_sim")
    ap.add_argument("--repo-id", default="local/so101_pick_place_sim")
    ap.add_argument("--task", default=TASK)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--videos", action="store_true",
                    help="store cameras as MP4 (needs torchcodec + FFmpeg; default on Windows: images)")
    ap.add_argument("--viewer", action="store_true", help="show the viewer in real time (slow)")
    args = ap.parse_args()

    cfg = load_robot_config(args.config)
    root = resolve_path(args.root)
    if root.exists():
        if not args.overwrite:
            raise SystemExit(f"{root} exists; pass --overwrite to replace it")
        shutil.rmtree(root)
    rec = DemoRecorder(RecordingConfig(repo_id=args.repo_id, root=root, use_videos=args.videos))
    robot = MuJoCoRobot.from_config(cfg)
    task = PickPlaceTask(robot)
    home = np.asarray(cfg.poses["home"])
    log, kept, seed, t0 = [], 0, args.start_seed, time.perf_counter()
    try:
        with RealtimeViewer(robot, headless=not args.viewer) as viewer:
            ctrl = ArmController.from_config(robot, cfg, on_step=viewer)
            ik = None
            while kept < args.episodes:
                task.reset(home, seed=seed)
                ctrl.sync()
                ik = ik or ArmIK(robot, "main", cfg.arms[0].model)
                entry = {"seed": seed, "layout": task.layout}
                try:
                    leader, plan = expert_leader(task, ik, robot.get_state().arms["main"].positions, home,
                                                 cfg.safety, ctrl.dt, clock=lambda: robot.data.time)
                except PlanningError as exc:
                    log.append({**entry, "kept": False, "reason": str(exc)})
                    seed += 1
                    continue
                leader.connect()
                sel = CommandSelector(ctrl, "main", {"expert": leader})
                sel.select("expert")
                viewer.restart_clock()
                ticks = int(np.ceil((leader.duration + 0.5) / ctrl.dt))
                grasped = False
                for _ in range(ticks):
                    rec.step(robot, sel, args.task)
                    grasped = grasped or task.state().grasped
                s = task.state()
                ok = s.in_box and not any("source_error" in e for e in sel.log.events)
                entry.update(kept=ok, frames=rec.frames_in_episode, grasped=grasped, in_box=s.in_box,
                             max_ik_error_mm=max(p["ik_error_mm"] for p in plan))
                (rec.save_episode if ok else rec.discard_episode)()
                kept += ok
                log.append(entry)
                print(f"seed {seed}: {'kept' if ok else 'discarded'} ({entry['frames']} frames, "
                      f"in_box={s.in_box}) [{kept}/{args.episodes}]")
                seed += 1
    except (ViewerClosed, KeyboardInterrupt):
        print("stopped early; finalizing the episodes saved so far")
    finally:
        rec.finalize()
        robot.close()
    summary = {"repo_id": args.repo_id, "root": str(root), "task": args.task, "episodes_kept": kept,
               "attempts": len(log), "frames": int(sum(e.get("frames", 0) for e in log if e.get("kept"))),
               "wall_seconds": round(time.perf_counter() - t0, 1), "episodes": log}
    (root / "recording_log.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n{kept} episodes ({summary['frames']} frames) from {len(log)} attempts in "
          f"{summary['wall_seconds']} s -> {Path(root).relative_to(PROJECT_ROOT) if root.is_relative_to(PROJECT_ROOT) else root}")


if __name__ == "__main__":
    main()
