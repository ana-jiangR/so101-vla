# so101-vla

SO-ARM101 robot arm in MuJoCo simulation, with a reusable control stack and
leader→follower teleoperation. The goal is to drive it with a
Vision-Language-Action model (SmolVLA) and, eventually, a two-arm "duck"
robot with a head camera. See [design.md](design.md) for the full brief.

| Stage | Status |
|-------|--------|
| 1. Low-level control + teleoperation in MuJoCo | **done** (sim) |
| 2. Pretrained SmolVLA in simulation | **done**: pipeline runs closed loop; zero-shot results in [docs/stage2.md](docs/stage2.md) |
| 3. Demonstrations + fine-tuning | in progress (simulation): fine-tuned SmolVLA places the cube in 4/20 unseen layouts; see [docs/stage3.md](docs/stage3.md) |

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
runner was the reference for Stage 2's name, camera and unit matching.

Stage 2 uses LeRobot's official tabletop scene and its success test, and runs SmolVLA
only through LeRobot's own loading and `lerobot-eval` inference steps. Where each
piece comes from is listed in [docs/stage2.md](docs/stage2.md).

## Roadmap

1. **Stage 1: low-level control and teleop in MuJoCo (done).** Official model, a
   `RobotInterface` with a MuJoCo backend, safety filter, trajectories, tracking logs,
   wrist-camera rendering, LeRobot-format observations, a virtual leader → simulated
   follower, and a command selector with one active source.
2. **Stage 2: pretrained SmolVLA in simulation (done).** LeRobot's tabletop scene (one
   cube, green box as target) with top, front, wrist and an egocentric head camera;
   `lerobot/smolvla_base` loaded with LeRobot's APIs; offline inference, then closed-loop
   chunk execution through the same `CommandSelector` → `SafetyFilter` path; zero-shot
   evaluation at three levels (inference / control / task success). Dependencies are in
   the `vla` extra.
3. **Stage 3: demonstrations and fine-tuning (in progress, simulation first).** Zero-shot
   `smolvla_base` does not solve the task, and the official docs say it must be
   fine-tuned on the target robot and task. Plan:
   1. record demonstrations in this MuJoCo scene, using a scripted expert policy
      (many episodes, automatically), the on-screen sliders, or later a real leader
      arm driving the simulated follower (already supported by `CommandSelector`)
   2. save them as a LeRobotDataset (`TeleopLog` / `PolicySource` logs already hold
      the per-tick state, actions and camera frames)
   3. fine-tune `smolvla_base` with `lerobot-train` (this also gives correct SO-101
      normalization stats), and compare against the Stage 2 zero-shot results with
      the same `m6`/`m7` scripts; try 10 / 50 / 100–200 demonstrations
   4. with hardware: real leader → real follower demonstrations and sim-to-real tests

   Only step 4 needs the physical arms. Nothing in Stages 1–2 depends on training code.

Bimanual "duck" robot: every API is keyed by arm name, and the two-arm config builds
and is tested. Observations use LeRobot's `left_`/`right_` keys.

## Setup

| Platform | Use it for | Guide |
|---|---|---|
| **Windows 11** (tested) | everything: simulation, teleop, SmolVLA inference, recording, short fine-tuning runs | [docs/setup-windows.md](docs/setup-windows.md) |
| **Linux** (GPU workstation or WSL2) | long fine-tuning runs (Stage 3); no Windows workarounds, video datasets, large GPUs | [docs/setup-linux.md](docs/setup-linux.md) |

Install only what each stage needs (all versions pinned in `uv.lock`):

| Command | Installs | GPU |
|---|---|---|
| `uv sync` | Stage 1: MuJoCo, numpy, PyYAML | no |
| `uv sync --extra vla` | + Stage 2: PyTorch 2.11 (CUDA 12.8), `lerobot[smolvla]` 0.6.1 | for inference (~0.9 GiB) |
| `uv sync --extra train` | + Stage 3: `lerobot[training]` (datasets, PyAV, torchcodec), `mink` | for fine-tuning (~3.6 GB at batch 8) |

### Quick start (Windows PowerShell)

```powershell
uv sync
uv run pytest
uv run python scripts/m2_joint_demo.py      # scripted joint/gripper/trajectory demo
uv run python scripts/m3_teleop_demo.py     # drive the arm with on-screen sliders

uv run python scripts/m4_tabletop_scene.py       # Stage 2 scene + camera images (no GPU)
uv sync --extra vla                              # Stage 2: torch (CUDA) + LeRobot
uv run python scripts/m5_offline_inference.py    # SmolVLA prediction only, robot doesn't move
uv run python scripts/m6_vla_closed_loop.py      # SmolVLA drives the arm (viewer)
uv run python scripts/m7_zero_shot_eval.py       # zero-shot evaluation, headless (~10 min)

uv sync --extra train                            # Stage 3: dataset + training tools
uv run python scripts/m8_record_demos.py --episodes 10   # scripted expert -> LeRobotDataset
uv run python scripts/m9_finetune.py --episodes 10 --steps 2000
```

### Quick start (Linux)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
sudo apt-get install -y git ffmpeg                # FFmpeg: video datasets
git clone https://github.com/ana-jiangR/sol101-vla.git so101-vla && cd so101-vla
uv sync --extra train
export MUJOCO_GL=egl                              # only on a server without a display
uv run pytest -q
uv run python scripts/m8_record_demos.py --episodes 50 --videos
uv run python scripts/m9_finetune.py --episodes 50 --steps 20000 --batch-size 64 --save-freq 5000
```

The Linux steps follow the official uv, PyTorch, LeRobot and MuJoCo instructions but have not
been run in this repo yet; see [docs/setup-linux.md](docs/setup-linux.md) for details, copying
datasets/checkpoints between machines, and WSL2.

## Layout

```text
assets/robotstudio_so101/  MuJoCo Menagerie SO-101 (Apache-2.0, unmodified, commit pinned by LeRobot)
assets/lerobot_tabletop/   LeRobot's SO-101 tabletop scene (Apache-2.0) + our one-cube variant
assets/scenes/             our Stage 1 scene; arms are attached in code (as LeRobot's env does)
configs/robot/             arms, mounts, rates, safety limits, named poses, extra cameras
configs/policy/            SmolVLA checkpoint, camera mapping, stats mode
src/so101_vla/
  conventions.py           joint order, units, LeRobot <-> radians conversion (one place)
  config.py                YAML -> dataclasses
  robot/                   RobotInterface (backend contract) + MuJoCoRobot
  control/                 SafetyFilter, trajectories, ArmController (numpy only), mink IK (Stage 3)
  teleop/                  virtual leaders, CommandSelector (one active source)
  sim/                     scene composition, cameras, real-time viewer, tabletop task metrics
  policies/                SmolVLA via LeRobot, action-chunk validator, policy as command source,
                           scripted pick-and-place expert
  recording.py             demonstrations -> LeRobotDataset (lerobot-record's schema and order)
  evaluation/              closed-loop episodes, three-level results
  utils/                   PNG output
scripts/                   m1_* … m9_* milestone demos, asset fetchers
tests/                     headless pytest suite
docs/                      conventions, setup (Windows, Linux), Stage 2/3 reports, sample_logs/
```

## Architecture

```text
 command sources (LeRobot-style get_action(): degrees, gripper 0..100)
   SliderLeader | TrajectoryLeader | lerobot SO101Leader (future) | PolicySource (SmolVLA, Stage 2)
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

Pretrained SmolVLA as a command source (needs `uv sync --extra vla`):

```python
from so101_vla.policies import PolicySource
from so101_vla.policies.smolvla import SmolVLAPolicy, load_smolvla_config
from so101_vla.teleop import CommandSelector

cfg = load_robot_config("configs/robot/so101_tabletop.yaml")
robot = MuJoCoRobot.from_config(cfg)
ctrl = ArmController.from_config(robot, cfg)
ctrl.reset("home")
policy = SmolVLAPolicy(load_smolvla_config("configs/policy/smolvla_base.yaml"))
source = PolicySource.for_robot(robot, policy, "Pick up the cube.")   # validated action chunks
sel = CommandSelector(ctrl, "main", {"policy": source})
sel.select("policy")
sel.run(10.0)            # re-observes and predicts a new chunk every 50 actions
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

## Stage 2 results (simulation, zero-shot `smolvla_base`)

18 episodes (the design's 3 instructions × 3 seeds × 2 normalization conditions), details in
[docs/stage2.md](docs/stage2.md):

- **Level 1, inference: 18/18.** Valid (50, 6) chunks; ~0.8 s per chunk, 0.9 GiB VRAM on an RTX 4060 laptop GPU.
- **Level 2, robot control: 9/18.** Every action was validated and clipped to the joint limits. The 9 failures are the
  "checkpoint stats" runs, where the arm reached toward the cube but pressed on the table.
- **Level 3, task success: 0/18.** The cube was touched in 5 episodes but never grasped.
- The checkpoint's saved normalization stats don't apply in LeRobot 0.6.1, and they describe SO-100
  (old calibration) data. The official docs present `smolvla_base` as a base model for fine-tuning.

## Stage 3 results so far (simulation, fine-tuned on scripted demonstrations)

Scripted IK expert: 50/52 layouts recorded (every kept episode places the cube in the box).
Fine-tuned with `lerobot-train` on the laptop GPU (RTX 4060, 8 GB) and evaluated on unseen
layouts ([docs/stage3.md](docs/stage3.md)):

| | zero-shot | 10 ep × 2k steps | 50 ep × 5k steps | **50 ep × 20k steps, batch 16** |
|---|---|---|---|---|
| training time | – | 34 min | 74 min | 317 min (overnight) |
| L2 control (no table contact) | 0/9 | 7/10 | 9/10 | 19/20 |
| grasped / lifted | 0 / 0 | 1 / 0 | 2 / 1 | 4 / 4 |
| **L3: cube placed in the box and released** | **0/9** | **0/10** | **0/10** | **4/20** |

The 4 placements (seeds 2, 10, 13, 19) were checked from the logged cube position (resting on
the box floor, released). The policy is stochastic: a second run of the same checkpoint
grasped 8/20 and placed at least 1. Failures are missed grasps; once grasped, the cube was
always carried and placed.

Sample outputs from these runs are in [docs/sample_logs/](docs/sample_logs/).

## Known limitations

- Simulated servo gains are not calibrated to real STS3215 servos (per the upstream note).
- Real hardware isn't connected yet. The interfaces match LeRobot's `SO101Leader` / `SO101Follower`, but
  the real-leader → sim-follower and real → real paths are untested, and real-arm calibration offsets
  against the sim zero haven't been checked.
- The safety layer checks joint limits only. Contacts with the floor/table are measured (tests, Stage 2
  evaluation) but not prevented; there is no self-collision analysis or motion planning, and safe engage
  moves in a straight line in joint space.
- Velocity/acceleration limits (1.5 rad/s, 6 rad/s²) are conservative defaults, not measured servo limits.
- `reset()` teleports the robot, which only works in simulation. Real hardware must use `move_to()`.
- The bimanual config exists, but teleop currently routes one source to one arm (`key_prefix` can pick
  one arm out of a LeRobot `bi_so_leader` action). Driving both arms needs one selector for both arms.
- Real calibration only matches the sim zero if each joint is swept through its full range during
  `lerobot-calibrate`. The real gripper's 0..100 spans the recorded range, not the sim's. In degrees mode,
  `wrist_roll` may be offset by about 2.8° between real and sim (asymmetric MJCF range). See docs/conventions.md.
- Zero-shot SmolVLA does not solve the task (see docs/stage2.md). `smolvla_base`'s saved normalization stats
  don't apply in LeRobot 0.6.1 (legacy keys) and describe SO-100 old-calibration data; there is no documented
  mapping to SO-101 units. Fine-tuning on SO-101 data (Stage 3) is the documented route.
- Inference is synchronous (sim time pauses during the ~0.8 s per chunk); LeRobot's asynchronous inference
  (needed on real hardware for smooth motion) is not used yet.
- The policy sees the `top`, `wrist` and `front` cameras (SmolVLA's pretraining order). From the start pose the
  wrist camera does not see the cube, and in the top view the cube is ~10 px wide and partly hidden by the arm.
  The egocentric `head` camera is rendered but not fed to the policy by default (`camera_map` in
  `configs/policy/smolvla_base.yaml`).
