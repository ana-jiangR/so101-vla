"""Milestone 6: pretrained SmolVLA drives the simulated SO-101 in closed loop (Task 2.6).

Chunks of predicted actions go through the action validator, the command selector
(safe engage), and the Stage 1 safety filter before reaching the MuJoCo actuators.
The environment is re-observed after each chunk. The episode ends on success (cube
in box), on a policy/validation error (the arm then holds), or at the timeout.

    uv run python scripts/m6_vla_closed_loop.py                      # with viewer, real time
    uv run python scripts/m6_vla_closed_loop.py --headless --seed 2 --stats-mode checkpoint_action_stats
"""

from __future__ import annotations

import argparse
import json
import time

from so101_vla.config import PROJECT_ROOT, load_robot_config
from so101_vla.control import ArmController
from so101_vla.evaluation import EpisodeConfig, run_episode
from so101_vla.policies import PolicySource
from so101_vla.policies.smolvla import SmolVLAPolicy, load_smolvla_config
from so101_vla.robot.mujoco_robot import MuJoCoRobot
from so101_vla.sim.tabletop import PickPlaceTask
from so101_vla.sim.viewer import RealtimeViewer, ViewerClosed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--robot-config", default="configs/robot/so101_tabletop.yaml")
    ap.add_argument("--policy-config", default="configs/policy/smolvla_base.yaml")
    ap.add_argument("--stats-mode", choices=["as_shipped", "checkpoint_action_stats"], default=None)
    ap.add_argument("--instruction", default="Pick up the red cube and place it in the green box.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=30.0, help="episode timeout (sim time)")
    ap.add_argument("--replan-every", type=int, default=None, help="actions per chunk to execute (default: all)")
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    cfg = load_robot_config(args.robot_config)
    pcfg = load_smolvla_config(args.policy_config, stats_mode=args.stats_mode)
    print(f"loading {pcfg.checkpoint} ({pcfg.stats_mode}) ...")
    policy = SmolVLAPolicy(pcfg)
    robot = MuJoCoRobot.from_config(cfg)
    task = PickPlaceTask(robot)
    out = PROJECT_ROOT / "outputs" / f"vla_closed_loop_{pcfg.stats_mode}_seed{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    source = PolicySource.for_robot(robot, policy, args.instruction, replan_every=args.replan_every)
    try:
        with RealtimeViewer(robot, headless=args.headless) as viewer:
            ctrl = ArmController.from_config(robot, cfg, on_step=viewer)
            viewer.restart_clock()
            result = run_episode(robot, ctrl, task, source,
                                 EpisodeConfig(args.instruction, args.seed, args.seconds), out)
    except (ViewerClosed, KeyboardInterrupt):
        print("stopped early; no episode result saved")
        return
    finally:
        robot.close()
    print(json.dumps({k: v for k, v in result.items() if k != "layout"}, indent=1))
    print(f"logs saved to {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
