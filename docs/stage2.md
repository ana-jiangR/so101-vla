# Stage 2: pretrained SmolVLA in simulation

Pretrained `lerobot/smolvla_base` drives the simulated SO-101 with no fine-tuning and
no demonstrations. Every step follows an official source; the table below says which.

| Step | What we use | Source |
|---|---|---|
| Scene, cube, target box, cameras | LeRobot's `so101-tabletop-mujoco` scene (one-cube variant) | [EnvHub env](https://huggingface.co/lerobot/so101-tabletop-mujoco) (Apache-2.0), `assets/lerobot_tabletop/PROVENANCE.md` |
| Reset randomization, "cube in box" | LeRobot env's `DEFAULTS["randomize"]` and `cubes_in_box` | its `env.py` |
| Loading the model | `SmolVLAPolicy.from_pretrained` + `make_pre_post_processors` | [model card](https://huggingface.co/lerobot/smolvla_base) |
| Observation → batch → action | `preprocess_observation`, `observation["task"]`, preprocessor, `predict_action_chunk`, postprocessor, with `lerobot-eval`'s processor overrides | `lerobot/scripts/lerobot_eval.py` (`rollout`, `main`), LeRobot 0.6.1 |
| Camera order | image 1 = top, 2 = wrist, 3 = side | [SmolVLA paper](https://arxiv.org/abs/2506.01844), Sec. 3.2 |
| Chunk execution | play the chunk, re-observe when it runs out | paper Sec. 4.3; `SmolVLAPolicy.select_action` |
| Units (state/action) | LeRobot `so101_follower` degrees, gripper 0..100 | LeRobot env `LEROBOT_SIM["semantics"]` |
| Camera placement maths | camera looks along −Z, +X right, +Y up; `xyaxes` | [MuJoCo modeling docs](https://mujoco.readthedocs.io/en/stable/modeling.html) |
| Reference integration | name/camera/unit matching, refusing mismatched units | Strands Robots `policy/lerobot/contract.py`, `units/lerobot.py` |

Our own code is only the glue: [`policies/smolvla.py`](../src/so101_vla/policies/smolvla.py)
(wraps the official calls), [`action_adapter.py`](../src/so101_vla/policies/action_adapter.py)
(checks), [`policy_source.py`](../src/so101_vla/policies/policy_source.py) (chunk queue as a
command source), [`sim/tabletop.py`](../src/so101_vla/sim/tabletop.py) (task measurements),
[`evaluation/rollout.py`](../src/so101_vla/evaluation/rollout.py) (episodes and levels).

## Pipeline

```text
MuJoCo cameras (top, wrist, front; 256x256)  +  joint state (LeRobot degrees)  +  instruction
      |  RobotInterface.get_observation()
      v
LeRobot preprocess_observation -> rename to camera1/2/3 -> tokenizer -> normalizer
      v
SmolVLA predict_action_chunk  (50 actions, 10 flow-matching steps, ~0.8 s on an RTX 4060 laptop GPU)
      v
LeRobot postprocessor (unnormalizer)
      v
ActionChunkValidator: shape (50, 6), NaN/inf -> reject (hold); range/jump report
      v
PolicySource (one action per 30 Hz tick; re-observe after the chunk)
      v
CommandSelector (one active source, safe engage) -> LeRobotUnits -> radians
      v
SafetyFilter (soft limits, velocity/acceleration) -> MuJoCoRobot (hard-limit check, actuators, 7 x 1/210 s physics)
```

Physics (210 Hz), control (30 Hz) and inference (once per 50-action chunk) are separate
rates. Inference is synchronous: simulated time does not advance while the model runs,
as in the paper's real-robot evaluation.

## Cameras: what the policy sees

World frame: origin at the arm base on the table top, +x forward, +y left, z up. The cube
starts ~23 cm in front of the base (seeded offsets up to ±6 cm in x, ±14 cm in y), the box
to the left.

| Camera | Position | View | Source | Policy input |
|---|---|---|---|---|
| `top` | (0.20, 0, 0.48) m | straight down | LeRobot scene | `camera1` (top) |
| `main/wrist_cam` | on the gripper | out past the jaws | Menagerie arm | `camera2` (wrist) |
| `front` | (0.49, 0, 0.20) m | back at the arm, 30° down | LeRobot scene | `camera3` (side) |
| `head` | (−0.10, 0.16, 0.45) m | down/forward at the table | ours (`configs/robot/so101_tabletop.yaml`) | not used by default |

Each time a chunk is requested, MuJoCo renders the policy's cameras at 256×256 RGB; LeRobot
converts them and SmolVLA pads them to 512×512. The images, the 6 joint angles and the
instruction are the whole input. The cube's true position is used only by the evaluation,
never given to the policy.

At the start pose (seed 0, images in `sample_logs/stage2_offline_inference_*/`): the front
view shows the cube, box and arm clearly; in the top view the cube is ~10 px wide next to the
gripper and partly hidden; the wrist camera sees the table, the wall and a jaw, but not the
cube. The head camera sees cube, box and arm, but is not one of SmolVLA's pretrained view
types, so it is left out of the default mapping.

## Task 2.3: what the checkpoint expects (verified)

From `config.json`, the processor files and the loaded objects (LeRobot 0.6.1, checkpoint
revision `5e8d12a`):

| | `smolvla_base` |
|---|---|
| Cameras | `observation.images.camera1/2/3`, each 3×256×256, float in [0, 1] (LeRobot's `preprocess_observation`); resized with padding to 512×512 inside the policy |
| State | `observation.state`, 6 values, padded to 32 inside the model |
| Action | 6 values, padded to 32 inside the model |
| Chunk | `chunk_size` = `n_action_steps` = 50 |
| Normalization | MEAN_STD for state and action, identity for images |
| Language | SmolVLM2 tokenizer, max 48 tokens, padded |
| Device / precision | CUDA, float32 (`use_amp: false`); 0.9 GiB VRAM; ~0.8 s per chunk (2.6–2.8 s for the first call) |
| Backbone | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, first 16 layers, 450 M parameters total |

**Finding: the saved normalization stats do not apply.** The only stats file,
`policy_preprocessor_step_5_normalizer_processor.safetensors`, holds *action* mean/std for
three SO-100 setups, under legacy keys `so100.buffer.action.mean` (and `so100-red…`,
`so100-blue…`). There are no `observation.state` stats. LeRobot 0.6.1 loads them as
features named `so100.buffer.action`, and its normalizer looks up `action` and
`observation.state`, finds neither, and passes values through unchanged
(`normalize_processor.py`, `_apply_transform`). So with the official pipeline as shipped,
the model sees raw degrees as its "normalized" state and its normalized outputs
(roughly −3…+6) are passed on as if they were degrees. Community reports match:
missing stats for zero-shot `smolvla_base` ([lerobot#2351](https://github.com/huggingface/lerobot/issues/2351),
[blog](https://huggingface.co/blog/Beegbrain/daily-robotics-june-2)); the official docs
say the base model is meant to be fine-tuned.

**Finding: the action space is SO-100, not SO-101.** The saved action means
(`shoulder_lift` ≈ 120–126°, `elbow_flex` ≈ 110–125°, `wrist_roll` ≈ −27…−107°) are
impossible in −100..100 and far from the SO-101 "new calibration" zero we use (mid-range).
They come from SO-100 community datasets recorded with LeRobot's older calibration. There
is no documented mapping from that calibration to the SO-101 one, so we don't invent one.

We therefore evaluate two explicitly labelled conditions (`stats_mode` in
`configs/policy/smolvla_base.yaml`):

- `as_shipped`: LeRobot's pipeline exactly as published (stats silently unused).
- `checkpoint_action_stats`: the checkpoint's own `so100` action stats, applied through
  the normalizer's documented `stats` override, to the action and, as there are no
  state stats, also to the state.

Neither is "correct" for this robot; both are reported.

## Task 2.4: observation and action mapping

| Item | Mapping |
|---|---|
| Joint order | `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper` everywhere (MJCF = LeRobot = checkpoint's 6-D state/action) |
| Units | state and actions in LeRobot `so101_follower` degrees, gripper 0..100 (`conventions.LeRobotUnits`), the same as LeRobot's env |
| Gripper | 0 = closed (MJCF lower limit), 100 = open |
| Normalization | see above (`stats_mode`) |
| Chunk timing | 50 actions at 30 Hz = 1.67 s per chunk, then re-observe (`--replan-every` to replan sooner) |
| Control rate | 30 Hz (LeRobot SO-101 standard) |
| Cameras | `camera1` = `top`, `camera2` = `main/wrist_cam`, `camera3` = `front` (paper order: top, wrist, side). The egocentric `head` camera exists and can be mapped instead in the policy config |
| Padding / unused dims | state and action are padded to 32 inside the model and unpadded by LeRobot (`_get_action_chunk`); we only ever see 6 values |
| Safety | no raw output is a joint target: validator → selector engage → safety filter → backend |

## Task 2.7: zero-shot evaluation (2026-11-03)

`uv run python scripts/m7_zero_shot_eval.py`: the design's three instructions × seeds 0, 1, 2
× both stats modes, 30 s (900 control ticks) per episode, cameras top/wrist/front at 256×256,
whole 50-action chunks. Full results: [sample_logs/stage2_zero_shot_eval/](sample_logs/stage2_zero_shot_eval/).

| stats mode | instruction | L1 inference | L2 control | L3 success | touched cube | grasped | min gripper–cube (cm, mean) | arm–table contact (ticks of 900, mean) |
|---|---|---|---|---|---|---|---|---|
| as_shipped | Move the gripper toward the cube. | 3/3 | 3/3 | 0/3 | 0/3 | 0/3 | 12.2 | 0 |
| as_shipped | Pick up the cube. | 3/3 | 3/3 | 0/3 | 0/3 | 0/3 | 12.2 | 0 |
| as_shipped | Place the cube in the target area. | 3/3 | 3/3 | 0/3 | 0/3 | 0/3 | 12.2 | 0 |
| checkpoint_action_stats | Move the gripper toward the cube. | 3/3 | 0/3 | 0/3 | 1/3 | 0/3 | 3.8 | 819 |
| checkpoint_action_stats | Pick up the cube. | 3/3 | 0/3 | 0/3 | 1/3 | 0/3 | 4.5 | 794 |
| checkpoint_action_stats | Place the cube in the target area. | 3/3 | 0/3 | 0/3 | 3/3 | 0/3 | 2.8 | 699 |

**Level 1 (inference): 18/18.** Every chunk had shape (50, 6) and finite values.
Inference took 0.80–0.87 s per chunk (mean per episode) on the RTX 4060 laptop GPU.

**Level 2 (robot control): 9/18, all in `as_shipped`.** No chunk was rejected, nothing
crashed, and every out-of-range action was clipped by the safety filter. The 9
`checkpoint_action_stats` episodes fail Level 2 only because the arm stayed in contact with
the table for most of the episode (641–846 of 900 ticks); joint limits held throughout.

**Level 3 (task success): 0/18.** The cube was never grasped, lifted or placed in the box.

Observed failure modes:

- **`as_shipped`: the arm stands up and stays there.** The first chunk asks for a pose up to
  61° from the start (the model's near-zero normalized outputs read as degrees), so the
  safe-engage step and the rate limits take the arm toward the all-zero pose (pointing
  straight up). It never approaches the cube: the closest gripper–cube distance (11.0–13.8
  cm) is the starting distance, and the cube never moved. The gripper action is negative
  (below "closed") on 272–786 of 900 steps and is clipped. The instruction makes no visible
  difference, which is consistent with the state input being far outside what the model saw.
- **`checkpoint_action_stats`: the arm reaches toward the cube but drives into the table.**
  Each chunk starts within a few degrees of the current pose (the stats put the model's
  outputs in a plausible range), and the gripper gets to 1.6–6.0 cm of the cube. It touched
  the cube in 5 of 9 episodes and pushed it up to 10.9 cm ("Move…", seed 2; in "Place…",
  seed 2 it ended pushed 7.6 cm against the box's outer wall), but never closed on it. The forearm and wrist press on the table for
  most of the episode; joint targets beyond the limits (mostly `shoulder_lift`, `wrist_flex`,
  gripper) were clipped. "Place…" touched the cube 3/3 vs 1/3 for the others, but with 3
  seeds per cell that difference means little.

Why this is expected, and what it does not show:

- The checkpoint's action space is SO-100 (old calibration), its state has no stats, and its
  camera views come from real SO-100 setups, not this simulated scene. The official docs
  describe it as a base model to fine-tune; we did no fine-tuning, as Stage 2 requires.
- `checkpoint_action_stats` applies SO-100 action stats to SO-101 joint angles. That puts
  the numbers in the right range but not in the right calibration, so "reaching toward the
  cube" here should not be read as understanding the scene.
- What Stage 2 does show: the full pipeline (observation → LeRobot preprocessing → SmolVLA →
  validation → safety filter → MuJoCo actuators → re-observe) runs closed loop on Windows,
  every predicted action is checked, and the arm never exceeded its joint limits.

Next steps (Stage 3, not done here): record SO-101 demonstrations in this scene with the
Stage 1 teleop stack, fine-tune `smolvla_base` on them (which also gives correct SO-101
stats), and compare against these zero-shot numbers. An existing SO-101 task checkpoint
could serve as a separate baseline (design.md Task 2.7, optional).

## How to run

```powershell
uv sync --extra vla                                      # torch (CUDA 12.8) + lerobot[smolvla] 0.6.1
uv run python scripts/m4_tabletop_scene.py               # scene + all camera images (no GPU)
uv run python scripts/m5_offline_inference.py            # Task 2.5: inference only, robot never moves
uv run python scripts/m5_offline_inference.py --stats-mode checkpoint_action_stats
uv run python scripts/m6_vla_closed_loop.py              # Task 2.6: closed loop, viewer in real time
uv run python scripts/m7_zero_shot_eval.py               # Task 2.7: all conditions, headless
```

The first `m5` run downloads the checkpoint (~1 GB) and the SmolVLM2 tokenizer into the
Hugging Face cache. Outputs go to `outputs/`.
