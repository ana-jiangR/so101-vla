"""Milestone 10: offline check of a policy on the REAL SO-101 follower; the arm is never moved.

Reads the real follower's joint positions and wrist-camera image exactly like LeRobot's
``SO101Follower.get_observation`` (calibrated, ``so101_follower`` degrees, gripper 0..100),
runs SmolVLA on them, and reports what it *would* command: range vs the joint limits,
distance from the current pose, jumps. Nothing is ever sent to the motors, and torque is
not touched (``SO101Follower.connect`` would enable torque; we only connect the bus and
the camera). Inputs and outputs are saved.

The fine-tuned checkpoint was trained on 3 simulated views (top, wrist, front). With one camera,
it is fed as the view given by ``--slot`` (default ``camera3``, the front view) and the others are
left out; SmolVLA allows missing views (``prepare_images`` needs at least one). Expect a large
sim-to-real gap.

    uv run python scripts/m10_real_offline_check.py --camera-index 0 --slot camera3 --rotate 180

Zero-shot MolmoAct2-SO100/101 (``--policy molmoact2``, configs/policy/molmoact2_so101.yaml) takes two
views, in any order; it was trained on real SO-100/101 data. With one camera the same 640x480
frame fills both slots (as the community molmoact2-so101 ``--scene-only`` mode does); add a second
camera (e.g. the wrist camera) with ``--camera2-index``. ``--slot`` / ``--checkpoint`` are SmolVLA-only.

    uv run python scripts/m10_real_offline_check.py --policy molmoact2 --camera-index 0
    uv run python scripts/m10_real_offline_check.py --policy molmoact2 --camera-index 0 --camera2-index 1
"""

from __future__ import annotations

import argparse
import json
import time

import cv2
import numpy as np

from so101_vla.config import PROJECT_ROOT, load_robot_config
from so101_vla.conventions import JOINT_NAMES, LEROBOT_FEATURE_NAMES, LeRobotUnits
from so101_vla.policies import ActionChunkValidator, InvalidChunk
from so101_vla.policies.smolvla import SmolVLAConfig, SmolVLAPolicy
from so101_vla.robot.mujoco_robot import MuJoCoRobot
from so101_vla.utils.images import save_png

DEFAULT_CK = "outputs/train/smolvla_so101_pick_place_ep50_steps20000_bs16/checkpoints/020000/pretrained_model"


def square_resize(img: np.ndarray, size: int) -> np.ndarray:
    """Center-crop to a square (keeps the aspect ratio), then resize; RGB uint8."""
    h, w = img.shape[:2]
    s = min(h, w)
    y, x = (h - s) // 2, (w - s) // 2
    return cv2.resize(img[y:y + s, x:x + s], (size, size), interpolation=cv2.INTER_AREA)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", default="COM12")
    ap.add_argument("--robot-id", default="so101_follower", help="calibration id (lerobot-calibrate --robot.id)")
    ap.add_argument("--camera-index", type=int, default=0, help="OpenCV index of the camera")
    ap.add_argument("--slot", choices=["camera1", "camera2", "camera3"], default="camera3",
                    help="which policy view this camera stands in for: camera1=top, camera2=wrist, "
                         "camera3=front/side (default; place it ~50 cm in front of the base, ~20 cm up)")
    ap.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0,
                    help="rotate the camera image (e.g. 180 for an upside-down mounted camera)")
    ap.add_argument("--checkpoint", default=DEFAULT_CK)
    ap.add_argument("--policy", choices=["smolvla", "molmoact2"], default="smolvla")
    ap.add_argument("--policy-config", default="configs/policy/molmoact2_so101.yaml", help="molmoact2 only")
    ap.add_argument("--camera2-index", type=int, default=None,
                    help="molmoact2 only: second camera (e.g. wrist); default: reuse the first camera")
    ap.add_argument("--instruction", default="Pick up the red cube and place it in the green box.")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    from lerobot.cameras import Cv2Rotation
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    cameras = {"wrist": OpenCVCameraConfig(index_or_path=args.camera_index, width=640, height=480, fps=30,
                                           rotation=Cv2Rotation({270: -90}.get(args.rotate, args.rotate)))}
    if args.policy == "molmoact2" and args.camera2_index is not None:
        cameras["cam2"] = OpenCVCameraConfig(index_or_path=args.camera2_index, width=640, height=480, fps=30)
    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.robot_id, cameras=cameras))
    if not robot.calibration:
        raise SystemExit(f"no calibration for id {args.robot_id!r}; run lerobot-calibrate first")
    # Read-only: bus + cameras only. No configure(), no torque change, no send_action().
    robot.bus.connect()
    for cam in robot.cameras.values():
        cam.connect()
    try:
        for _ in range(10):  # let auto-exposure settle
            for cam in robot.cameras.values():
                cam.read()
            time.sleep(0.03)
        state = robot.bus.sync_read("Present_Position")
        frames = {name: np.asarray(cam.read()) for name, cam in robot.cameras.items()}
        torque = robot.bus.sync_read("Torque_Enable", normalize=False)
    finally:
        for cam in robot.cameras.values():
            cam.disconnect()
        robot.bus.disconnect(disable_torque=False)

    state_lr = np.array([state[j] for j in JOINT_NAMES], dtype=np.float64)
    frame = frames["wrist"]
    img = square_resize(frame, 256)
    print(f"real state (deg, gripper 0..100): {dict(zip(JOINT_NAMES, state_lr.round(1).tolist(), strict=True))}")
    print(f"torque enabled per motor (untouched): {torque}")

    # Joint limits of the simulated model (same degrees convention) for the range checks.
    sim = MuJoCoRobot.from_config(load_robot_config("configs/robot/so101_tabletop.yaml"))
    lim = sim.joint_limits("main")
    sim.close()
    units = LeRobotUnits(lim.lower, lim.upper)
    validator = ActionChunkValidator(units)
    lo, hi = validator.lower_lr, validator.upper_lr

    joints = dict(zip(LEROBOT_FEATURE_NAMES, state_lr.tolist(), strict=True))
    if args.policy == "molmoact2":
        from so101_vla.policies.molmoact2 import MolmoAct2Policy, load_molmoact2_config

        cmap = {"cam0": "wrist", "cam1": "cam2" if "cam2" in frames else "wrist"}
        policy = MolmoAct2Policy(load_molmoact2_config(args.policy_config, camera_map=cmap))
        print(f"camera {args.camera_index} -> cam0; {'camera ' + str(args.camera2_index) if 'cam2' in frames else 'same camera'} -> cam1")
        obs = {**joints, **frames}  # 640x480 RGB, as the checkpoint's LeRobot example
    else:
        policy = SmolVLAPolicy(SmolVLAConfig(checkpoint=args.checkpoint, revision=None, camera_map={args.slot: "wrist"}))
        print(f"camera {args.camera_index} -> policy view {args.slot}")
        obs = {**joints, "wrist": img}
    chunks, times = [], []
    for _ in range(args.repeats):
        policy.reset()
        chunks.append(policy.predict_chunk(obs, args.instruction))
        times.append(policy.last_inference_ms)
    chunk = chunks[0]
    report, error = None, None
    try:
        report = validator.validate(chunk, units.from_lerobot(state_lr))
    except InvalidChunk as exc:
        error = str(exc)

    print(f"\ninstruction: {args.instruction!r}; inference {', '.join(f'{t:.0f}' for t in times)} ms")
    print(f"\n{'joint':14s} {'lo':>7s} {'hi':>7s} {'state':>7s} {'a[0]':>7s} {'a[-1]':>7s} {'min':>7s} {'max':>7s}  oor")
    for j, n in enumerate(JOINT_NAMES):
        flag = "  <- state outside sim range" if not lo[j] <= state_lr[j] <= hi[j] else ""
        print(f"{n:14s} {lo[j]:7.1f} {hi[j]:7.1f} {state_lr[j]:7.1f} {chunk[0, j]:7.1f} {chunk[-1, j]:7.1f} "
              f"{chunk[:, j].min():7.1f} {chunk[:, j].max():7.1f}  {report.out_of_range_steps[n] if report else '-'}{flag}")
    spread = np.std(np.array(chunks), axis=0).mean(axis=0)
    print("sampling spread over repeats:", spread.round(2))
    print("validation:", "REJECTED: " + error if error else
          f"ok; first action vs current pose (deg): {report.first_step_distance}; "
          f"largest step-to-step change: {report.max_step_change}")

    out = PROJECT_ROOT / "outputs" / f"real_offline_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True)
    save_png(out / "wrist_raw.png", np.asarray(frame))
    save_png(out / "wrist_256.png", img)
    if "cam2" in frames:
        save_png(out / "cam2_raw.png", frames["cam2"])
    np.savez_compressed(out / "sample.npz", state_lerobot=state_lr, chunks=np.array(chunks), image=img,
                        joint_names=np.array(JOINT_NAMES), limits_lerobot=np.array([lo, hi]))
    (out / "sample.json").write_text(json.dumps({
        "port": args.port, "robot_id": args.robot_id, "camera_index": args.camera_index, "policy_slot": args.slot,
        "policy": args.policy, "checkpoint": args.checkpoint if args.policy == "smolvla" else policy.describe(),
        "instruction": args.instruction, "inference_ms": times,
        "state": dict(zip(JOINT_NAMES, state_lr.tolist(), strict=True)), "torque_enabled": torque,
        "validation": report.to_dict() if report else {"rejected": error},
        "sampling_spread": dict(zip(JOINT_NAMES, spread.round(3).tolist(), strict=True)),
    }, indent=2, default=str), encoding="utf-8")
    print(f"\nsaved to {out.relative_to(PROJECT_ROOT)} (the arm was not moved)")


if __name__ == "__main__":
    main()
