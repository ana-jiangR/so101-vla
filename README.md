# so101-vla

SO-ARM101 robot arm in MuJoCo simulation, with a reusable control stack and
leader→follower teleoperation. The goal is to drive it with a
Vision-Language-Action model (SmolVLA) and, eventually, a two-arm "duck"
robot with a head camera. See [design.md](design.md) for the full brief.

| Stage | Status |
|-------|--------|
| 1. Low-level control + teleoperation in MuJoCo | **done** (sim) |
| 2. Pretrained SmolVLA in simulation | not started |
| 3. Demonstrations + fine-tuning | future |

Where an official source exists, we follow it. The robot model, its commit, the
units, the 30 Hz rate, and the home pose all match LeRobot's official
[SO-101 MuJoCo env](https://huggingface.co/lerobot/so101-tabletop-mujoco). The
virtual leaders implement LeRobot's `Teleoperator` interface. The optional
`max_relative_target` safeguard and the `left_`/`right_` two-arm key names come
from LeRobot's follower code.

What we reuse rather than rebuild:
- **Official model:** the MuJoCo Menagerie SO-101.
- **LeRobot's conversions and scene-building approach**, copied from its sim env.
- **LeRobot's real-arm classes later:** a real `SO101Leader` plugs into `CommandSelector`
  unchanged. Pass `norm="degrees"` if it was created with `use_degrees=True`, or
  `norm="range_m100_100"` for LeRobot's default.
- **Our own code where nothing official exists:** stock LeRobot has no sim robot
  type for `lerobot-teleoperate`, so the teleop loop for a sim follower is ours.

[Strands Robots](https://github.com/strands-labs/robots) covers similar ground:
a `Robot`/`Simulation` API on the same Menagerie SO-101 model. We read its docs
and source. We follow its handling of LeRobot units: both `degrees` and
`range_m100_100` are supported, plus the calibrated-travel ↔ joint-limits
assumption. We also follow its "hold on fault" behaviour. We kept our own small,
tested stack instead of depending on the framework: it already handles two arms,
teleop and the 30 Hz LeRobot timing, and Strands brings in agents, a mesh network
and multiple simulator backends that this project doesn't need. Its LeRobot policy
runner is the reference for Stage 2.

## Roadmap

1. **Stage 1: low-level control and teleop in MuJoCo (done).** Official model, a
   `RobotInterface` with a MuJoCo backend, safety filter, trajectories, tracking logs,
   wrist-camera rendering, LeRobot-format observations, a virtual leader → simulated
   follower, and a command selector with one active source.
2. **Stage 2: pretrained SmolVLA in simulation (next).** Tabletop scene with a cube,
   target and head camera; load `lerobot/smolvla_base` with LeRobot's APIs; offline
   inference first, then closed-loop control with action chunks through the same
   `CommandSelector` → `SafetyFilter` path; zero-shot evaluation at three levels
   (inference / control / task success). Dependencies go in the `vla` extra.
3. **Stage 3: demonstrations and fine-tuning (future, documented only).** Record
   teleop episodes (the `TeleopLog` arrays already hold leader/commanded/measured per
   tick) as a LeRobotDataset, fine-tune SmolVLA, compare against zero-shot, then real
   leader/follower data. Nothing in Stages 1–2 depends on training code.

Bimanual "duck" robot: every API is keyed by arm name, and the two-arm config builds
and is tested. Observations use LeRobot's `left_`/`right_` keys.

## Quick start (Windows)

```powershell
uv sync
uv run pytest
uv run python scripts/m2_joint_demo.py      # scripted joint/gripper/trajectory demo
uv run python scripts/m3_teleop_demo.py     # drive the arm with on-screen sliders
```

See [docs/setup-windows.md](docs/setup-windows.md) for details.

## Layout

```text
assets/robotstudio_so101/  MuJoCo Menagerie SO-101 (Apache-2.0, unmodified, commit pinned by LeRobot)
assets/scenes/             our scenes; arms are attached in code (as LeRobot's env does)
configs/robot/             arms, mounts, rates, safety limits, named poses
src/so101_vla/
  conventions.py           joint order, units, LeRobot <-> radians conversion (one place)
  config.py                YAML -> dataclasses
  robot/                   RobotInterface (backend contract) + MuJoCoRobot
  control/                 SafetyFilter, trajectories, ArmController (numpy only)
  teleop/                  virtual leaders, CommandSelector (one active source)
  sim/                     scene composition, real-time viewer
scripts/                   m1_* m2_* m3_* milestone demos, asset fetcher
tests/                     headless pytest suite
docs/                      conventions (joint/unit findings), setup, sample_logs/
```

## Architecture

```text
 command sources (LeRobot-style get_action(): degrees, gripper 0..100)
   SliderLeader | TrajectoryLeader | lerobot SO101Leader (future) | VLA policy (Stage 2)
        |
 CommandSelector ── one active source; safe engage on start/switch
        |
 conventions.LeRobotUnits ── degrees or range_m100_100 / gripper 0..100 -> radians
        |
 ArmController ── SafetyFilter: NaN/shape check, soft limits, velocity + accel limits,
        |                       optional max_relative_target (vs. measured, as LeRobot)
        |          TrackingLog: commanded vs measured every tick
 RobotInterface ── MuJoCoRobot: hard-limit check -> data.ctrl -> physics substeps
                └─ RealRobot (future, same interface)
```

- **Multi-arm from the start.** Per-arm values are dicts keyed by arm name;
  `configs/robot/so101_bimanual.yaml` builds and is tested.
- **`data.ctrl` = joint target in radians** (position actuators, checked when the model loads).
  Normal motion only goes through actuators. State is set directly only by `reset()`.
- **Two layers of validation.** The controller clips (or rejects) to soft limits and
  rate-limits every command. The backend still rejects anything outside the hard limits.

Joint order, units, gripper direction, and calibration are documented in
[docs/conventions.md](docs/conventions.md).

## Using the API

```python
from so101_vla.config import load_robot_config
from so101_vla.control import ArmController
from so101_vla.robot.mujoco_robot import MuJoCoRobot

cfg = load_robot_config("configs/robot/so101_single.yaml")
robot = MuJoCoRobot.from_config(cfg)
ctrl = ArmController.from_config(robot, cfg)

ctrl.reset("rest")
ctrl.move_to("home")                          # smooth min-jerk move
ctrl.move_joint("main", "shoulder_pan", 0.5)  # one joint [rad]
ctrl.close_gripper("main")
state = ctrl.get_state()                      # state.arms["main"].positions / .velocities

img = robot.render_camera("main/wrist_cam", 640, 480)       # (480, 640, 3) uint8 RGB
obs = robot.get_observation(cameras=["main/wrist_cam"])     # LeRobot format:
# {"shoulder_pan.pos": deg, ..., "gripper.pos": 0..100, "main/wrist_cam": image}
```

Teleoperation from any LeRobot-style source:

```python
from so101_vla.teleop import CommandSelector, TrajectoryLeader

leader = TrajectoryLeader.hold([20, -60, 60, 35, 0, 80], clock=lambda: robot.data.time)  # LeRobot units
leader.connect()
sel = CommandSelector(ctrl, "main", {"leader": leader})
sel.select("leader")     # follower aligns slowly, then tracks
sel.run(5.0)
sel.log.save("outputs/my_teleop", ctrl.dt)
```

## Stage 1 results (simulation)

- **Model:** Menagerie SO-101 loads with 6 joints, 6 position actuators, and `wrist_cam`. Physics matches
  LeRobot's env (implicitfast, 7 × 1/210 s substeps at 30 Hz).
- **Joint control:** each joint moves ±0.4 rad individually and settles within 0.01 rad. In the full joint
  demo (about 49 s of sim time), the largest tracking error is under 0.01 rad, with no clipping and no contacts.
- **Gripper:** the full range is reachable, and the jaw gap goes from 45.5 mm (closed) to 87.1 mm (open).
- **Teleop (Task 1.6):** the virtual leader drives the follower with an RMS leader→follower error ≤ 0.008 rad.
  The follower lags by one 30 Hz control period (33 ms). Compute takes about 0.4 ms per tick against a
  33 ms budget. Switching sources re-engages without jumps.

Sample outputs from these runs are in [docs/sample_logs/](docs/sample_logs/).

## Known limitations

- Simulated servo gains are not calibrated to real STS3215 servos (per the upstream note).
- Real hardware isn't connected yet. The interfaces match LeRobot's `SO101Leader` / `SO101Follower`, but
  the real-leader → sim-follower and real → real paths are untested, and real-arm calibration offsets
  against the sim zero haven't been checked.
- Collision checking covers the floor only. There is no self-collision analysis or motion planning; safe
  engage moves in a straight line in joint space.
- Velocity/acceleration limits (1.5 rad/s, 6 rad/s²) are conservative defaults, not measured servo limits.
- `reset()` teleports the robot, which only works in simulation. Real hardware must use `move_to()`.
- The bimanual config exists, but teleop currently routes one source to one arm (`key_prefix` can pick
  one arm out of a LeRobot `bi_so_leader` action). Driving both arms needs one selector for both arms.
- Real calibration only matches the sim zero if each joint is swept through its full range during
  `lerobot-calibrate`. The real gripper's 0..100 spans the recorded range, not the sim's. In degrees mode,
  `wrist_roll` may be offset by about 2.8° between real and sim (asymmetric MJCF range). See docs/conventions.md.
- Only the arm's wrist camera exists so far. The head/overview camera and the tabletop scene are Stage 2 (Task 2.1–2.2).
