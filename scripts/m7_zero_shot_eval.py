"""Milestone 7: zero-shot evaluation of pretrained SmolVLA on the tabletop task (Task 2.7).

Runs every (stats mode x instruction x seed) episode headless and reports the
three levels separately (inference / robot control / task success) plus progress
measures. Success is only counted when the cube is detected in the box.

    uv run python scripts/m7_zero_shot_eval.py                       # 2 modes x 3 instructions x 3 seeds
    uv run python scripts/m7_zero_shot_eval.py --seeds 0 1 2 3 4 --seconds 40
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

import numpy as np

from so101_vla.config import PROJECT_ROOT, load_robot_config
from so101_vla.control import ArmController
from so101_vla.evaluation import EpisodeConfig, run_episode
from so101_vla.policies import PolicySource
from so101_vla.policies.smolvla import SmolVLAPolicy, load_smolvla_config
from so101_vla.robot.mujoco_robot import MuJoCoRobot
from so101_vla.sim.tabletop import PickPlaceTask

# design.md Task 2.7
INSTRUCTIONS = ["Move the gripper toward the cube.", "Pick up the cube.", "Place the cube in the target area."]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--robot-config", default="configs/robot/so101_tabletop.yaml")
    ap.add_argument("--policy-config", default="configs/policy/smolvla_base.yaml")
    ap.add_argument("--stats-modes", nargs="+", default=["as_shipped", "checkpoint_action_stats"])
    ap.add_argument("--instructions", nargs="+", default=INSTRUCTIONS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--seconds", type=float, default=30.0)
    args = ap.parse_args()

    cfg = load_robot_config(args.robot_config)
    out = PROJECT_ROOT / "outputs" / f"zero_shot_eval_{time.strftime('%Y%m%d_%H%M%S')}"
    results = []
    for mode in args.stats_modes:
        pcfg = load_smolvla_config(args.policy_config, stats_mode=mode)
        policy = SmolVLAPolicy(pcfg)
        for instruction in args.instructions:
            for seed in args.seeds:
                robot = MuJoCoRobot.from_config(cfg)
                ctrl = ArmController.from_config(robot, cfg)
                task = PickPlaceTask(robot)
                ep_dir = out / mode / f"{instruction[:-1].lower().replace(' ', '_')}_seed{seed}"
                res = run_episode(robot, ctrl, task, PolicySource.for_robot(robot, policy, instruction),
                                  EpisodeConfig(instruction, seed, args.seconds), ep_dir)
                robot.close()
                res["stats_mode"] = mode
                results.append(res)
                p = res["progress"]
                print(f"[{mode}] {instruction!r} seed {seed}: L1={res['level1_inference']} "
                      f"L2={res['level2_control']} L3={res['level3_success']} | min dist "
                      f"{p['min_gripper_to_cube_m'] * 100:.1f} cm, touched={p['touched_cube']}, "
                      f"grasped={p['grasped_cube']}, clipped ticks={res['ticks_position_clipped']}, "
                      f"table contact ticks={res['ticks_arm_touching_table']}")
        del policy

    groups = defaultdict(list)
    for r in results:
        groups[(r["stats_mode"], r["instruction"])].append(r)
    summary = []
    for (mode, instr), rs in groups.items():
        summary.append({
            "stats_mode": mode, "instruction": instr, "episodes": len(rs),
            "level1_inference": sum(r["level1_inference"] for r in rs),
            "level2_control": sum(r["level2_control"] for r in rs),
            "level3_success": sum(r["level3_success"] for r in rs),
            "touched_cube": sum(r["progress"]["touched_cube"] for r in rs),
            "grasped_cube": sum(r["progress"]["grasped_cube"] for r in rs),
            "min_gripper_to_cube_cm_mean": round(100 * float(np.mean([r["progress"]["min_gripper_to_cube_m"] for r in rs])), 1),
            "clipped_ticks_mean": round(float(np.mean([r["ticks_position_clipped"] or 0 for r in rs])), 1),
            "table_contact_ticks_mean": round(float(np.mean([r["ticks_arm_touching_table"] for r in rs])), 1),
            "inference_ms_mean": round(float(np.mean([r["inference_ms_mean"] for r in rs if r["inference_ms_mean"]])), 0),
        })
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({"config": vars(args), "summary": summary, "episodes": results},
                                                 indent=2, default=str), encoding="utf-8")
    lines = ["| stats mode | instruction | L1 inference | L2 control | L3 success | touched | grasped | "
             "min dist (cm) | clipped ticks | table-contact ticks |", "|" + "---|" * 10]
    for s in summary:
        n = s["episodes"]
        lines.append(f"| {s['stats_mode']} | {s['instruction']} | {s['level1_inference']}/{n} | {s['level2_control']}/{n} | "
                     f"{s['level3_success']}/{n} | {s['touched_cube']}/{n} | {s['grasped_cube']}/{n} | "
                     f"{s['min_gripper_to_cube_cm_mean']} | {s['clipped_ticks_mean']} | {s['table_contact_ticks_mean']} |")
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nresults saved to {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
