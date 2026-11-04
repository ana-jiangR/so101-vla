"""Task 1.6: virtual leader -> simulated SO-101 follower teleoperation.

Leaders (all produce LeRobot-style actions: degrees, gripper 0..100):
  sliders  on-screen joint sliders (default; needs a display)
  sweep    scripted "operator" moving every joint at once (works headless)
  replay   replays the leader commands of a saved teleop log

--switch-demo (with sweep/replay) also exercises source switching:
leader -> "policy" stand-in (holds a different pose) -> leader, each switch
re-engaging safely. Logs go to outputs/teleop_<timestamp>/.

Usage:
    uv run python scripts/m3_teleop_demo.py
    uv run python scripts/m3_teleop_demo.py --leader sweep --headless --switch-demo
    uv run python scripts/m3_teleop_demo.py --leader replay --log outputs/teleop_.../teleop.npz
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from datetime import datetime

import numpy as np

from so101_vla.config import PROJECT_ROOT, load_robot_config
from so101_vla.control import ArmController
from so101_vla.conventions import GRIPPER_INDEX, LEROBOT_FEATURE_NAMES, to_lerobot
from so101_vla.robot.mujoco_robot import MuJoCoRobot
from so101_vla.sim.viewer import RealtimeViewer, ViewerClosed
from so101_vla.teleop import CommandSelector, SliderLeader, TrajectoryLeader

ARM = "main"
# LeRobot units: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll [deg], gripper [0..100]
# Around "home": stays >= 2.5 cm above the floor (checked over 60 s), gripper 5..55.
SWEEP_AMPLITUDE = [30.0, 10.0, 10.0, 15.0, 45.0, 25.0]
SWEEP_PERIOD = [6.0, 7.0, 5.0, 4.5, 8.0, 3.0]
POLICY_STANDIN_POSE = [-25.0, -45.0, 50.0, 60.0, 20.0, 80.0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="configs/robot/so101_single.yaml")
    p.add_argument("--leader", choices=["sliders", "sweep", "replay"], default="sliders")
    p.add_argument("--log", help="teleop.npz to replay (--leader replay)")
    p.add_argument("--duration", type=float, default=20.0, help="seconds (sweep/replay)")
    p.add_argument("--switch-demo", action="store_true")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--fps", type=float, default=None,
                   help="teleop/control rate in Hz (default: config, 30 = LeRobot recording rate; "
                        "lerobot-teleoperate defaults to 60)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.leader == "sliders" and args.headless:
        p.error("--leader sliders needs a display")

    cfg = load_robot_config(args.config)
    if args.fps is not None:
        cfg = dataclasses.replace(cfg, control_hz=args.fps)
    robot = MuJoCoRobot.from_config(cfg)
    sim_clock = lambda: robot.data.time  # noqa: E731 - leaders follow sim time, not wall time

    with RealtimeViewer(robot, headless=args.headless) as viewer:
        ctrl = ArmController.from_config(robot, cfg, on_step=viewer)
        # Sliders start wherever the follower is; the sweep starts at "home", like an
        # operator who matches the follower before engaging; a replay starts where
        # the recorded session's leader started, so it reproduces it from tick one.
        if args.leader == "replay" and args.log:
            ctrl.reset(np.load(args.log)["leader"][0])
        else:
            ctrl.reset("rest" if args.leader == "sliders" else "home")
        viewer.restart_clock()

        hard = robot.joint_limits(ARM)
        g_lo, g_hi = hard.lower[GRIPPER_INDEX], hard.upper[GRIPPER_INDEX]
        home_lr = to_lerobot(np.asarray(cfg.poses["home"]), g_lo, g_hi)

        if args.leader == "sliders":
            soft = ctrl.limits(ARM)
            lo_lr, hi_lr = to_lerobot(soft.lower, g_lo, g_hi), to_lerobot(soft.upper, g_lo, g_hi)
            start_lr = to_lerobot(ctrl.filters[ARM].last_target, g_lo, g_hi)
            leader = SliderLeader(
                initial=dict(zip(LEROBOT_FEATURE_NAMES, start_lr.round(1), strict=True)),
                ranges={k: (float(a), float(b)) for k, a, b in zip(LEROBOT_FEATURE_NAMES, lo_lr, hi_lr, strict=True)},
                home=dict(zip(LEROBOT_FEATURE_NAMES, home_lr.round(1), strict=True)),
            )
        elif args.leader == "sweep":
            leader = TrajectoryLeader.sweep(home_lr, SWEEP_AMPLITUDE, SWEEP_PERIOD, args.duration, clock=sim_clock)
        else:
            if not args.log:
                p.error("--leader replay needs --log PATH")
            leader = TrajectoryLeader.from_teleop_log(args.log, clock=sim_clock)
            args.duration = leader.duration

        sources = {"leader": leader}
        if args.switch_demo:
            sources["policy"] = TrajectoryLeader.hold(POLICY_STANDIN_POSE, clock=sim_clock)
        selector = CommandSelector(ctrl, ARM, sources)
        for s in sources.values():
            s.connect()

        print(f"Teleop: {args.leader} leader -> simulated follower at {1 / ctrl.dt:.0f} Hz "
              "(follower first aligns slowly to the leader)")
        try:
            selector.select("leader")
            if args.leader == "sliders":
                print("Move the sliders. Close the slider window (or the viewer) to stop.")
                while not leader.closed:
                    selector.tick()
            elif args.switch_demo:
                third = args.duration / 3
                selector.run(third)
                print(f"t={robot.data.time:.1f}s switch -> policy stand-in")
                selector.select("policy")
                selector.run(third)
                print(f"t={robot.data.time:.1f}s switch -> leader")
                selector.select("leader")
                selector.run(third)
            else:
                selector.run(args.duration)
        except (ViewerClosed, KeyboardInterrupt):
            print("\nStopped early; saving logs of what ran so far.")
        finally:
            for s in sources.values():
                s.disconnect()

        mode = args.leader + ("_switch" if args.switch_demo else "")
        out = PROJECT_ROOT / "outputs" / f"teleop_{mode}_{datetime.now():%Y%m%d_%H%M%S}"
        selector.log.events.insert(0, {"mode": mode, "replayed_from": args.log} if args.log else {"mode": mode})
        selector.log.save(out, ctrl.dt)
        ctrl.log.save(out)
        summary = selector.log.summary(ctrl.dt)
        print(json.dumps({k: summary[k] for k in summary if k != "events"}, indent=2))
        print("events:", summary.get("events"))
        print(f"logs saved to {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
