"""Milestone 11: drive the REAL SO-101 follower through our safety stack, then run the policy.

Two modes:

  pose-test (default, no model): move slowly to the simulation's start pose (LeRobot env HOME),
      hold it so you can compare with the simulation (same joint directions?), then go back to
      where the arm started and release torque.
  policy: move to the start pose, then let SmolVLA drive the arm for ``--seconds`` through
      PolicySource -> CommandSelector (slow safe engage) -> SafetyFilter (slow speed limits,
      calibration-based joint limits) -> LeRobot SO101Follower (driver-level max_relative_target),
      then return to the starting pose.

Safety: keep a hand on the follower's power plug. Ctrl+C freezes the arm where it is; you are
then asked to support it before torque is released.

    uv run python scripts/m11_real_closed_loop.py                       # pose test, no model
    uv run python scripts/m11_real_closed_loop.py --mode policy --seconds 20

Zero-shot MolmoAct2-SO100/101 (``--policy molmoact2``): two views in any order; one camera fills both
slots unless ``--camera2-index`` adds a second (e.g. wrist). Inference takes ~2.5 s per 30-action
chunk on an 8 GB laptop GPU (int8); the arm holds still while it thinks, then plays 1 s of motion.
Run ``m10_real_offline_check.py --policy molmoact2`` first.

    uv run python scripts/m11_real_closed_loop.py --mode policy --policy molmoact2 --seconds 60
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from so101_vla.config import PROJECT_ROOT, SafetyConfig, load_robot_config
from so101_vla.control import ArmController
from so101_vla.conventions import JOINT_NAMES

DEFAULT_CK = "outputs/train/smolvla_so101_pick_place_ep50_steps20000_bs16/checkpoints/020000/pretrained_model"


def deg(q: np.ndarray) -> list[float]:
    return np.degrees(np.asarray(q)).round(1).tolist()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=["pose-test", "policy"], default="pose-test")
    ap.add_argument("--port", default="COM12")
    ap.add_argument("--robot-id", default="so101_follower")
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0)
    ap.add_argument("--slot", choices=["camera1", "camera2", "camera3"], default="camera3",
                    help="policy view the camera stands in for (camera3 = front)")
    ap.add_argument("--checkpoint", default=DEFAULT_CK)
    ap.add_argument("--policy", choices=["smolvla", "molmoact2"], default="smolvla")
    ap.add_argument("--policy-config", default="configs/policy/molmoact2_so101.yaml", help="molmoact2 only")
    ap.add_argument("--camera2-index", type=int, default=None,
                    help="molmoact2 only: second camera (e.g. wrist); default: reuse the first camera")
    ap.add_argument("--instruction", default="Pick up the red cube and place it in the green box.")
    ap.add_argument("--seconds", type=float, default=20.0, help="policy run time")
    ap.add_argument("--max-velocity", type=float, default=0.5, help="rad/s per joint (sim default 1.5)")
    ap.add_argument("--max-acceleration", type=float, default=2.0, help="rad/s^2 per joint (sim default 6)")
    ap.add_argument("--max-relative-target", type=float, default=5.0, help="LeRobot driver clamp [deg]")
    ap.add_argument("--smooth", type=float, default=0.3, help="EMA action smoothing for the policy (0 = off)")
    ap.add_argument("--hold", type=float, default=4.0, help="seconds to hold the start pose")
    ap.add_argument("--start", choices=["home", "current"], default="home",
                    help="policy mode: 'home' = first move to the simulation start pose (sim-trained models); "
                         "'current' = start from where the arm is (models trained on real demos that "
                         "started from the folded rest pose)")
    args = ap.parse_args()

    from lerobot.cameras import Cv2Rotation
    from lerobot.cameras.opencv import OpenCVCameraConfig

    from so101_vla.robot.lerobot_real import LeRobotSO101

    cams = {}
    if args.mode == "policy":
        cams["front"] = OpenCVCameraConfig(index_or_path=args.camera_index, width=640, height=480, fps=30,
                                           rotation=Cv2Rotation({270: -90}.get(args.rotate, args.rotate)))
        if args.policy == "molmoact2" and args.camera2_index is not None:
            cams["second"] = OpenCVCameraConfig(index_or_path=args.camera2_index, width=640, height=480, fps=30)
    robot = LeRobotSO101(args.port, args.robot_id, cameras=cams, max_relative_target=args.max_relative_target)

    policy = None
    if args.mode == "policy" and args.policy == "molmoact2":  # load before touching the arm (minutes)
        from so101_vla.policies.molmoact2 import MolmoAct2Policy, load_molmoact2_config
        print("loading MolmoAct2 ...")
        cmap = {"cam0": "front", "cam1": "second" if "second" in cams else "front"}
        policy = MolmoAct2Policy(load_molmoact2_config(args.policy_config, camera_map=cmap))
    elif args.mode == "policy":  # load the model before touching the arm (takes ~1 min)
        from so101_vla.policies.smolvla import SmolVLAConfig, SmolVLAPolicy
        print("loading the policy ...")
        policy = SmolVLAPolicy(SmolVLAConfig(checkpoint=args.checkpoint, revision=None,
                                             camera_map={args.slot: "front"}))

    safety = SafetyConfig(position_margin=(0.03,) * 5 + (0.0,), max_velocity=(args.max_velocity,) * 6,
                          max_acceleration=(args.max_acceleration,) * 6, on_out_of_range="clip")
    home = np.asarray(load_robot_config("configs/robot/so101_tabletop.yaml").poses["home"])
    out = PROJECT_ROOT / "outputs" / f"real_{args.mode.replace('-', '_')}_{time.strftime('%Y%m%d_%H%M%S')}"
    log: dict = {"args": vars(args), "events": []}

    print(f"connecting to {args.port} (goal = present position before torque on) ...")
    robot.connect_safely()
    start = robot.get_state().arms["main"].positions.copy()
    lim = robot.joint_limits("main")
    print(f"start pose  (deg): {dict(zip(JOINT_NAMES, deg(start), strict=True))}")
    print(f"limits      (deg): {dict(zip(JOINT_NAMES, zip(deg(lim.lower), deg(lim.upper), strict=True), strict=True))}")
    ctrl = ArmController(robot, safety, poses={"home": tuple(home), "start": tuple(start)})
    log["start_deg"] = deg(start)
    interrupted = False
    try:
        if args.mode == "pose-test" or args.start == "home":
            print(f"\n1) moving to the simulation start pose {dict(zip(JOINT_NAMES, deg(home), strict=True))} "
                  f"(max {args.max_velocity} rad/s) ...")
            ctrl.move_to("home", speed_scale=1.0)
            ctrl.hold(args.hold)
            reached = robot.get_state().arms["main"].positions
            err = np.degrees(np.abs(reached - home))
            print(f"   reached (deg): {dict(zip(JOINT_NAMES, deg(reached), strict=True))}")
            print(f"   error   (deg): {dict(zip(JOINT_NAMES, err.round(1).tolist(), strict=True))}")
            log["home_reached_deg"], log["home_error_deg"] = deg(reached), err.round(2).tolist()
        else:
            print("\n1) starting from the current pose (as in the real demonstrations)")
            ctrl.hold(1.0)
        if args.mode == "pose-test":
            print("   -> compare with the simulation's start pose: gripper over the table, pointing down/forward.")

        if args.mode == "policy":
            from so101_vla.policies import PolicySource
            from so101_vla.teleop import CommandSelector, EngageConfig

            src = PolicySource.for_robot(robot, policy, args.instruction, smoothing=args.smooth or None)
            sel = CommandSelector(ctrl, "main", {"policy": src}, EngageConfig(engage_velocity=0.3))
            sel.select("policy")
            print(f"\n2) policy running for {args.seconds:.0f} s: {args.instruction!r} (Ctrl+C to freeze)")
            t_end = time.monotonic() + args.seconds
            while time.monotonic() < t_end and sel.active is not None:
                sel.tick()
            log["policy_events"] = sel.log.events
            out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out / "policy.npz", **src.arrays())
            sel.log.save(out, ctrl.dt)
            print(f"   {len(src.chunks)} chunks, {len(src.executed)} actions executed")

        print("\n3) returning to the starting pose ...")
        ctrl.move_to("start", speed_scale=1.0)
        ctrl.hold(1.0)
    except KeyboardInterrupt:
        interrupted = True
        robot.hold_here()
        log["events"].append("interrupted: froze in place")
        print("\nFROZEN in place (torque on).")
    finally:
        out.mkdir(parents=True, exist_ok=True)
        ctrl.log.save(out)
        (out / "run.json").write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")
        if interrupted:
            try:
                input("Support the arm with your hand, then press ENTER to release torque ...")
            except (KeyboardInterrupt, EOFError):
                pass
        robot.close(release_torque=True)
        print(f"torque released; logs in {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
