"""Milestone 5: offline SmolVLA inference; the robot is never moved (Task 2.5).

1. reset the tabletop scene (seeded) and capture the policy's camera images
2. read the robot state (LeRobot units)
3. give a language instruction
4. run pretrained SmolVLA with LeRobot's own loading and pre/post-processing
5. print the predicted action chunk next to the current state
6. validate shape, NaN/inf, joint ranges and jumps
7. save inputs and outputs for debugging

    uv run python scripts/m5_offline_inference.py
    uv run python scripts/m5_offline_inference.py --stats-mode checkpoint_action_stats --seed 3
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from so101_vla.config import PROJECT_ROOT, load_robot_config
from so101_vla.control import ArmController
from so101_vla.conventions import JOINT_NAMES, LEROBOT_FEATURE_NAMES, LeRobotUnits
from so101_vla.policies import ActionChunkValidator, InvalidChunk
from so101_vla.policies.smolvla import SmolVLAPolicy, load_smolvla_config
from so101_vla.robot.mujoco_robot import MuJoCoRobot
from so101_vla.sim.tabletop import PickPlaceTask
from so101_vla.utils.images import save_png


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--robot-config", default="configs/robot/so101_tabletop.yaml")
    ap.add_argument("--policy-config", default="configs/policy/smolvla_base.yaml")
    ap.add_argument("--stats-mode", choices=["as_shipped", "checkpoint_action_stats"], default=None)
    ap.add_argument("--checkpoint", default=None, help="Hub id or local fine-tuned checkpoint (pretrained_model dir)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--instruction", default="Pick up the red cube and place it in the green box.")
    ap.add_argument("--repeats", type=int, default=3, help="inference runs (timing + sampling spread)")
    args = ap.parse_args()

    cfg = load_robot_config(args.robot_config)
    robot = MuJoCoRobot.from_config(cfg)
    ctrl = ArmController.from_config(robot, cfg)
    task = PickPlaceTask(robot)
    task.reset(np.asarray(cfg.poses["home"]), seed=args.seed)
    ctrl.sync()

    pcfg = load_smolvla_config(args.policy_config, checkpoint=args.checkpoint, stats_mode=args.stats_mode, device=args.device)
    print(f"loading {pcfg.checkpoint} ({pcfg.stats_mode}) on {pcfg.device} ...")
    policy = SmolVLAPolicy(pcfg)
    info = policy.describe()
    print(json.dumps({k: info[k] for k in ("input_features", "output_features", "chunk_size", "n_action_steps",
                                           "normalizer_stats_keys_loaded", "stats_applied_to")}, indent=1))

    w, h = pcfg.image_size
    obs = robot.get_observation(cameras=policy.camera_names, width=w, height=h)
    state_lr = np.array([obs[k] for k in LEROBOT_FEATURE_NAMES])
    q = robot.get_state().arms["main"].positions
    lim = robot.joint_limits("main")
    validator = ActionChunkValidator(LeRobotUnits(lim.lower, lim.upper))

    chunks, times, report, error = [], [], None, None
    for _ in range(args.repeats):
        policy.reset()
        chunks.append(policy.predict_chunk(obs, args.instruction))
        times.append(policy.last_inference_ms)
    chunk = chunks[0]
    try:
        report = validator.validate(chunk, q)
    except InvalidChunk as exc:
        error = str(exc)

    print(f"\ninstruction: {args.instruction!r}   observation time: {robot.data.time:.3f} s (sim)")
    print(f"inference: {', '.join(f'{t:.0f}' for t in times)} ms; chunk shape {chunk.shape}")
    lo, hi = validator.lower_lr, validator.upper_lr
    print(f"\n{'joint':14s} {'limit lo':>9s} {'limit hi':>9s} {'state':>8s} {'a[0]':>8s} {'a[-1]':>8s} "
          f"{'min':>8s} {'max':>8s}  out-of-range steps")
    for j, name in enumerate(JOINT_NAMES):
        oor = report.out_of_range_steps[name] if report else "-"
        print(f"{name:14s} {lo[j]:9.1f} {hi[j]:9.1f} {state_lr[j]:8.1f} {chunk[0, j]:8.2f} {chunk[-1, j]:8.2f} "
              f"{chunk[:, j].min():8.2f} {chunk[:, j].max():8.2f}  {oor}")
    spread = np.std(np.array(chunks), axis=0).mean(axis=0) if len(chunks) > 1 else np.zeros(6)
    print("sampling spread (std over repeats, mean over steps):", np.round(spread, 3))
    print("validation:", "REJECTED: " + error if error else
          f"ok (finite, shape {report.steps}x6); out of range: {report.any_out_of_range}; "
          f"first-step distance from current: {report.first_step_distance}")

    out = PROJECT_ROOT / "outputs" / f"offline_inference_{pcfg.stats_mode}_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True)
    for cam in policy.camera_names:
        save_png(out / f"input_{cam.replace('/', '_')}.png", obs[cam])
    np.savez_compressed(out / "sample.npz", state_lerobot=state_lr, state_rad=q, chunks=np.array(chunks),
                        joint_names=np.array(JOINT_NAMES), limits_lerobot=np.array([lo, hi]),
                        **{f"image_{c.replace('/', '_')}": obs[c] for c in policy.camera_names})
    (out / "sample.json").write_text(json.dumps({
        "instruction": args.instruction, "seed": args.seed, "layout": task.layout,
        "observation_time_s": robot.data.time, "inference_ms": times, "policy": info,
        "validation": report.to_dict() if report else {"rejected": error},
        "sampling_spread": dict(zip(JOINT_NAMES, spread.round(4).tolist(), strict=True)),
    }, indent=2), encoding="utf-8")
    print(f"\nsaved inputs/outputs to {out.relative_to(PROJECT_ROOT)} (robot was not moved)")
    robot.close()


if __name__ == "__main__":
    main()
