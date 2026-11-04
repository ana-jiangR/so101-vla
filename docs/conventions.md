# SO-101 simulation conventions

Read from the **compiled** MuJoCo model (MuJoCo 3.17,
`assets/robotstudio_so101/so101.xml`, MuJoCo Menagerie at commit `0059d43`).
That is the model and commit used by LeRobot's official SO-101 MuJoCo
environment, [lerobot/so101-tabletop-mujoco](https://huggingface.co/lerobot/so101-tabletop-mujoco).
Menagerie derives it from TheRobotStudio's `so101_new_calib.xml` and adds
gripper collision primitives, `implicitfast` integration, the default actuator
force range, and a wrist camera (see `assets/robotstudio_so101/README.md`).

Regenerate the raw numbers with:

```powershell
uv run python scripts/m1_inspect_model.py
```

## Joints

Canonical order everywhere in this repo (`so101_vla.conventions.JOINT_NAMES`).
It matches the MJCF, LeRobot's `so101_follower` / `so101_leader` motors (IDs 1–6),
and LeRobot's env.

| # | Joint | Range [rad] | Range [deg] |
|---|-------|-------------|-------------|
| 0 | `shoulder_pan`  | −1.920 … +1.920 | −110.0 … +110.0 |
| 1 | `shoulder_lift` | −1.745 … +1.745 | −100.0 … +100.0 |
| 2 | `elbow_flex`    | −1.690 … +1.690 | −96.8 … +96.8 |
| 3 | `wrist_flex`    | −1.658 … +1.658 | −95.0 … +95.0 |
| 4 | `wrist_roll`    | −2.744 … +2.841 | −157.2 … +162.8 |
| 5 | `gripper`       | −0.175 … +1.745 | −10.0 … +100.0 |

- All six are hinge joints, in radians. Joint dynamics (class `sts3215`): damping 0.60, frictionloss 0.052, armature 0.028.
- The base is welded to the world, so `nq = nv = 6` per arm.

## Actuators: what `data.ctrl` means

There is one `<position>` actuator per joint, with the same name and order.
`MuJoCoRobot` verifies these properties when it loads the model:

- transmission = its own joint, gear = 1
- force = kp·(ctrl − q) − kv·q̇, with **kp = 998.22**, **kv = 2.731**
- **force range ±2.94 N·m** (Menagerie default)
- ctrlrange equals the joint range

So **`ctrl` is the target joint angle in radians**. These gains are
not a 1:1 mapping of the real servo gains (upstream comment).

## Units: ours vs. LeRobot

| | This repo (internal) | LeRobot (`so101-joint-degrees-v1`) |
|---|---|---|
| 5 arm joints | radians | degrees = `rad2deg(q)`, no offset |
| gripper | radians | 0..100, linear over ctrlrange, 0 = closed |
| names | `shoulder_pan`, … | `shoulder_pan.pos`, … |

Conversions: `conventions.to_lerobot` / `from_lerobot` (and the `_dict`
variants). They are copied from LeRobot's env (`env.py: to_ctrl / joint_state`).
Unlike LeRobot, `from_lerobot` doesn't clip; clipping happens in the safety layer,
where it's counted.

### The two LeRobot normalization modes

LeRobot normalizes the arm joints in one of two modes (`MotorNormMode`). Many
community SO-100/101 datasets and checkpoints use the default, so we support both
(`conventions.LeRobotUnits`, `CommandSelector(norm=...)`, `get_observation(norm=...)`):

| `norm` | LeRobot setting | Arm joints | Gripper |
|---|---|---|---|
| `"degrees"` (our default) | `use_degrees=True`; LeRobot's MuJoCo env | `rad2deg(q)` | 0..100 |
| `"range_m100_100"` | LeRobot's default (`use_degrees=False`) | lower..upper limit → −100..100, linear | 0..100 |

`range_m100_100` assumes the real arm's calibrated travel spans the model's joint
limits. Strands Robots (`strands_robots/units/lerobot.py`) makes the same assumption.
Before using a dataset or checkpoint, check which mode it was recorded in: look
at its stats or its robot config. Mixing up the modes scales every arm joint wrongly.

**wrist_roll offset (degrees mode only).** On real hardware, LeRobot's degrees are
measured from the middle of the calibrated range. LeRobot's sim env uses
`rad2deg(q)` with no offset. These agree for every joint except `wrist_roll`, whose MJCF range is
asymmetric (−2.744…+2.841 rad, middle +0.0485 rad = +2.8°). So a real leader's
`wrist_roll = 0°` can correspond to about +2.8° in the sim. We keep the official sim-env
convention. When real hardware is connected, check this against a known pose.

## Gripper

- Lower limit (−0.1745 rad) = closed, upper limit (+1.7453 rad) = open.
  This was verified by measuring the jaw: the gap grows from 45.5 mm (closed) to 87.1 mm (open).
- Normalized opening (`ArmController.set_gripper`): 0 = closed, 1 = open, over
  the full ctrlrange (equivalent to LeRobot's 0..100 divided by 100).
- The gripper has no soft-limit margin (`position_margin[5] = 0`), so its full LeRobot
  range is reachable. Grip force is capped by the 2.94 N·m actuator limit.

## Calibration: how real-arm degrees line up with the sim

- **Sim (new calibration):** each joint's zero is the middle of its range. At all
  zeros the arm points straight out along +x.
- **Real arm (LeRobot):** `lerobot-calibrate` has you
  1. put every joint in the middle of its range (homing: that pose becomes encoder
     count 2047, half of 4096), then
  2. move each joint through its full range, recording `range_min` / `range_max`.

  Then in `MotorNormMode.DEGREES` (`lerobot/motors/motors_bus.py`):

  ```text
  degrees = (raw - (range_min + range_max) / 2) * 360 / 4095
  ```

  So **0° = the middle of the range recorded during calibration**, the same
  convention as the sim. The two only agree if the joint is moved through
  roughly the same range the MJCF uses. A lopsided or partial sweep during
  calibration shifts the real zero.
- **Gripper on real hardware** (`RANGE_0_100`): 0..100 spans the *recorded*
  `range_min..range_max` (inverted if `drive_mode` is set). In the sim it spans the
  MJCF ctrlrange (−10°…100°). So the same "gripper 50" is not necessarily the same
  physical opening on the real arm and in the sim. Check this with hardware.
- **Calibration is per arm** (stored by `id`). LeRobot's docs stress that leader and
  follower must be calibrated so they read the same values in the same pose. That's
  what makes leader → follower (and real → sim) mappings line up.

## Rates

- Control **30 Hz** (LeRobot's SO-101 standard; `control_hz`).
- Physics: the model's 0.005 s timestep, adjusted as in LeRobot's env so a control
  period is a whole number of steps: `n = round(1/(30·0.005)) = 7`, so the timestep is
  1/210 s. Solver settings (`implicitfast`, elliptic cones,
  impratio 10) are copied from the arm MJCF because the scene has no `<option>`.

## Frames, sites, cameras

- World: z up, floor at z = 0. Each arm base sits at its configured `mount`.
- `gripperframe` site: approximate tool center point (`MuJoCoRobot.ee_position`).
- `wrist_cam` camera on the `camera_mount` body (fovy 61°).
- Tabletop scene (Stage 2): `front` and `top` cameras from LeRobot's scene, plus cameras from the
  robot config (`cameras:`, e.g. `head`), placed by position + look-at target. MuJoCo cameras look
  along their −Z axis with +X right and +Y up in the image (MuJoCo modeling docs). The table top is
  at z = 0 and the arm base sits on it at the origin.
- Elements of each attached arm are prefixed `"<arm>/"`, e.g. `main/elbow_flex`.
  (LeRobot's single-arm env attaches with no prefix.)

## Named poses (`configs/robot/*.yaml`)

| Pose | rad | Source |
|------|-----|--------|
| `home` | `[0, −1.0472, 1.0472, 0.6109, 0, 0.4014]` | LeRobot env `HOME = (0, −60, 60, 35, 0, 30)` |
| `rest` | `[0, −1.6, 1.55, 0.75, 0, 0]` | ours (no official rest pose): folded |
| `zero` | `[0, 0, 0, 0, 0, 0]` | calibration zero |

None of these poses has contacts. From `home`, any single joint can move ±0.4 rad
without hitting the floor. From `rest`, raising `shoulder_lift` drives the forearm
into the floor.

## Collisions

- Menagerie adds box/sphere collision primitives on both jaws and the camera
  mount, with manipulation-tuned contact settings (condim 6, friction `1 5e-3 5e-4`).
- The base has no collision geometry. The floor can collide with the arm.
  Self-collision hasn't been checked systematically.
