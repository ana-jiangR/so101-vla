"""Milestone 13: zero-shot MolmoAct2-SO100/101 on the simulated SO-101 (quantized for 8 GB GPUs).

``offline``: render the scene, predict a few chunks, print them next to the state and
validate them; the robot is never moved. Run this first: it shows VRAM use, latency and
whether the joint-frame correction gives sensible targets.

``run``: closed loop (validator -> command selector -> safety filter -> MuJoCo), for each
instruction x seed, with the same three-level report as m7.

    uv run python scripts/m13_molmoact2_sim.py offline
    uv run python scripts/m13_molmoact2_sim.py offline --quantization nf4
    uv run python scripts/m13_molmoact2_sim.py run --seeds 0 --viewer          # watch one episode
    uv run python scripts/m13_molmoact2_sim.py run --seeds 0 1 2 --seconds 40  # headless eval
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from so101_vla.config import PROJECT_ROOT, load_robot_config
from so101_vla.control import ArmController
from so101_vla.conventions import JOINT_NAMES, LEROBOT_FEATURE_NAMES, LeRobotUnits
from so101_vla.policies import ActionChunkValidator, InvalidChunk, PolicySource
from so101_vla.policies.molmoact2 import MolmoAct2Policy, load_molmoact2_config
from so101_vla.robot.mujoco_robot import MuJoCoRobot
from so101_vla.sim.tabletop import PickPlaceTask
from so101_vla.utils.images import save_png

INSTRUCTIONS = ["Pick up the red cube and place it in the green box."]


def offline(args, cfg, policy) -> None:
    robot = MuJoCoRobot.from_config(cfg)
    ctrl = ArmController.from_config(robot, cfg)
    task = PickPlaceTask(robot)
    task.reset(np.asarray(cfg.poses["home"]), seed=args.seeds[0])
    ctrl.sync()
    w, h = policy.config.image_size
    obs = robot.get_observation(cameras=policy.camera_names, width=w, height=h)
    state_lr = np.array([obs[k] for k in LEROBOT_FEATURE_NAMES])
    q = robot.get_state().arms["main"].positions
    lim = robot.joint_limits("main")
    validator = ActionChunkValidator(LeRobotUnits(lim.lower, lim.upper))
    instruction = args.instructions[0]

    chunks, times = [], []
    for _ in range(args.repeats):
        policy.reset()
        chunks.append(policy.predict_chunk(obs, instruction))
        times.append(policy.last_inference_ms)
    chunk, report, error = chunks[0], None, None
    try:
        report = validator.validate(chunk, q)
    except InvalidChunk as exc:
        error = str(exc)

    print(f"\ninstruction: {instruction!r}")
    print(f"inference: {', '.join(f'{t:.0f}' for t in times)} ms (first includes warm-up); chunk {chunk.shape}")
    print(f"GPU memory after inference: {policy.gpu_memory_gb()}")
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
          f"ok; out of range: {report.any_out_of_range}; first-step distance: {report.first_step_distance}")

    out = PROJECT_ROOT / "outputs" / f"molmoact2_offline_{policy.config.quantization}_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True)
    for cam in policy.camera_names:
        save_png(out / f"input_{cam.replace('/', '_')}.png", obs[cam])
    np.savez_compressed(out / "sample.npz", state_lerobot=state_lr, state_rad=q, chunks=np.array(chunks))
    (out / "sample.json").write_text(json.dumps({
        "instruction": instruction, "seed": args.seeds[0], "layout": task.layout, "inference_ms": times,
        "policy": policy.describe(), "validation": report.to_dict() if report else {"rejected": error},
        "sampling_spread": dict(zip(JOINT_NAMES, spread.round(4).tolist(), strict=True)),
    }, indent=2, default=str), encoding="utf-8")
    print(f"\nsaved to {out.relative_to(PROJECT_ROOT)} (robot was not moved)")
    robot.close()


def run(args, cfg, policy) -> None:
    from so101_vla.evaluation import EpisodeConfig, run_episode

    out = PROJECT_ROOT / "outputs" / f"molmoact2_{policy.config.quantization}_eval_{time.strftime('%Y%m%d_%H%M%S')}"
    results = []
    for instruction in args.instructions:
        for seed in args.seeds:
            robot = MuJoCoRobot.from_config(cfg)
            task = PickPlaceTask(robot)
            source = PolicySource.for_robot(robot, policy, instruction, replan_every=args.replan_every,
                                            smoothing=args.smooth)
            ep_dir = out / f"{instruction.rstrip('.').lower().replace(' ', '_')}_seed{seed}"
            try:
                if args.viewer:
                    from so101_vla.sim.viewer import RealtimeViewer

                    with RealtimeViewer(robot, headless=False) as viewer:
                        ctrl = ArmController.from_config(robot, cfg, on_step=viewer)
                        viewer.restart_clock()
                        res = run_episode(robot, ctrl, task, source, EpisodeConfig(instruction, seed, args.seconds), ep_dir)
                else:
                    ctrl = ArmController.from_config(robot, cfg)
                    res = run_episode(robot, ctrl, task, source, EpisodeConfig(instruction, seed, args.seconds), ep_dir)
            finally:
                robot.close()
            results.append(res)
            p = res["progress"]
            print(f"{instruction!r} seed {seed}: L1={res['level1_inference']} L2={res['level2_control']} "
                  f"L3={res['level3_success']} | min dist {p['min_gripper_to_cube_m'] * 100:.1f} cm, "
                  f"touched={p['touched_cube']}, grasped={p['grasped_cube']}, "
                  f"inference {res['inference_ms_mean'] or float('nan'):.0f} ms")
    n = len(results)
    summary = {
        "episodes": n, "level3_success": sum(r["level3_success"] for r in results),
        "touched_cube": sum(r["progress"]["touched_cube"] for r in results),
        "grasped_cube": sum(r["progress"]["grasped_cube"] for r in results),
        "min_gripper_to_cube_cm_mean": round(100 * float(np.mean([r["progress"]["min_gripper_to_cube_m"] for r in results])), 1),
        "inference_ms_mean": round(float(np.mean([r["inference_ms_mean"] for r in results if r["inference_ms_mean"]]
                                                 or [np.nan])), 0),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({"config": vars(args), "policy": policy.describe(),
                                                 "summary": summary, "episodes": results}, indent=2, default=str),
                                      encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print(f"results saved to {out.relative_to(PROJECT_ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["offline", "run"])
    ap.add_argument("--robot-config", default="configs/robot/so101_tabletop.yaml")
    ap.add_argument("--policy-config", default="configs/policy/molmoact2_so101.yaml")
    ap.add_argument("--quantization", choices=["none", "int8", "nf4"], default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--cam0", default=None, help="robot camera for slot cam0 (default from config: front)")
    ap.add_argument("--cam1", default=None, help="robot camera for slot cam1 (default from config: main/wrist_cam)")
    ap.add_argument("--num-steps", type=int, default=None, help="flow-matching steps (checkpoint: 10)")
    ap.add_argument("--instructions", nargs="+", default=INSTRUCTIONS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--repeats", type=int, default=3, help="offline: inference runs (timing + sampling spread)")
    ap.add_argument("--seconds", type=float, default=30.0, help="run: episode timeout (sim time)")
    ap.add_argument("--replan-every", type=int, default=None, help="run: actions per chunk (default: all 30)")
    ap.add_argument("--smooth", type=float, default=None, help="run: EMA action smoothing alpha in (0, 1]")
    ap.add_argument("--viewer", action="store_true", help="run: show the MuJoCo viewer (real time)")
    args = ap.parse_args()

    cfg = load_robot_config(args.robot_config)
    pcfg = load_molmoact2_config(args.policy_config, quantization=args.quantization, device=args.device,
                                 num_steps=args.num_steps)
    if args.cam0 or args.cam1:
        cams = dict(pcfg.camera_map)
        cams.update({k: v for k, v in (("cam0", args.cam0), ("cam1", args.cam1)) if v})
        pcfg = load_molmoact2_config(args.policy_config, quantization=pcfg.quantization, device=pcfg.device,
                                     num_steps=args.num_steps, camera_map=cams)
    print(f"loading {pcfg.checkpoint} ({pcfg.quantization}) on {pcfg.device} ...")
    policy = MolmoAct2Policy(pcfg)
    print(json.dumps(policy.describe(), indent=1, default=str))
    (offline if args.mode == "offline" else run)(args, cfg, policy)


if __name__ == "__main__":
    main()
