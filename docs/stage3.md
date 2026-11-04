# Stage 3: demonstrations and fine-tuning (simulation first)

Zero-shot `smolvla_base` does not solve the tabletop task (Stage 2: 0/18). The official
docs describe it as a base model to fine-tune on your own robot and task, so Stage 3
records SO-101 demonstrations in the same MuJoCo scene, fine-tunes on them with
LeRobot's own trainer, and re-runs the Stage 2 evaluation. No physical hardware is
needed; real leader/follower data and sim-to-real are later steps.

| design.md Stage 3 step | Status |
|---|---|
| 1. Simulated teleoperation or an expert control policy | done: scripted IK expert (and Stage 1 slider teleop) |
| 2. Record demonstrations in MuJoCo | done: `scripts/m8_record_demos.py` |
| 3. Convert to LeRobotDataset format | done: recorded directly as a LeRobotDataset v3 |
| 4. Real-world demonstrations (leader–follower) | needs hardware |
| 5. Fine-tune pretrained SmolVLA | done: `scripts/m9_finetune.py` (wraps `lerobot-train`) |
| 6. Compare zero-shot and fine-tuned | see Results |
| 7. Data efficiency (10 / 50 / 100–200 demos) | see Results |
| 8. Sim-to-real transfer | needs hardware |

## Where each piece comes from

| Step | What we use | Source |
|---|---|---|
| Expert IK | `mink` differential IK (FrameTask + PostureTask, ConfigurationLimit, DAQP) on the Menagerie MJCF | [mink](https://github.com/kevinzakka/mink). LeRobot's own `RobotKinematics` (placo + URDF) would be first choice, but [placo](https://pypi.org/project/placo/) has no Windows wheels |
| Expert execution | joint trajectory played as a command source through `CommandSelector` → safety filter → actuators | Stage 1 stack (same path as teleop) |
| Dataset schema | `hw_to_dataset_features` over `SO101Follower`-style observation/action dicts | `lerobot/scripts/lerobot_record.py` |
| Recording loop | observe → act → `build_dataset_frame` + `add_frame(task=…)` → `save_episode` → `finalize` | `lerobot_record.py`, `LeRobotDataset.create` |
| Fine-tuning | `lerobot-train --policy.path=lerobot/smolvla_base --dataset.repo_id=… --batch_size … --steps …` | [SmolVLA docs](https://huggingface.co/docs/lerobot/smolvla) |
| Camera mapping in training | `--rename_map` top/wrist/front → camera1/2/3 | `lerobot/configs/train.py` (`rename_map`), SmolVLA paper camera order |
| Evaluation | the Stage 2 `m7` script and three levels, with `--checkpoint`; L3 = cube in the box (LeRobot test) **and released** | Stage 2 |

## Scripted expert

[`policies/scripted.py`](../src/so101_vla/policies/scripted.py) reads the cube and box poses
after the seeded reset (privileged simulator state, which an expert may use) and plans:

```text
pregrasp (8 cm above the cube, jaws open) -> descend (1.2 cm) -> close -> lift (9 cm)
-> above the box -> lower (6 cm) -> open -> retreat -> home
```

The gripper approaches straight down; the jaws close across the pair of cube faces closest
to perpendicular to the arm's reach direction. Each waypoint is solved with IK inside the
soft joint limits, and the waypoints are joined with Stage 1 min-jerk segments within the
configured velocity/acceleration limits. The result is played back as a `TrajectoryLeader`,
so it drives the follower exactly like a leader arm would. An episode is kept only if the
cube is detected in the box at the end (LeRobot env definition).

Prototype check (before recording): 30/30 seeds succeeded, ~27 s of simulated time each.

## Dataset

`scripts/m8_record_demos.py` → `outputs/datasets/so101_pick_place_sim` (LeRobotDataset v3,
local, not pushed to the Hub):

| Feature | Content |
|---|---|
| `observation.state` | 6 measured joint positions, `so101_follower` degrees, gripper 0..100 (`shoulder_pan.pos` … `gripper.pos`) |
| `observation.images.top` / `.wrist` / `.front` | 256×256 RGB per tick (stored as images, see Windows notes) |
| `action` | the joint target the follower received that tick, same units and names |
| `task` | "Pick up the red cube and place it in the green box." |
| fps | 30 |

Training seeds start at 1000; evaluation uses seeds 0–99, so no evaluation layout is in the
training data.

## Windows notes (LeRobot 0.6.1)

- **Video**: LeRobot decodes video with torchcodec, which on Windows needs a shared ("full-shared")
  FFmpeg build installed system-wide. Without it, the dataset is stored as images
  (`use_videos=False`), about 50 MB per episode. With FFmpeg installed, set
  `RecordingConfig(use_videos=True)` for much smaller datasets.
- **`--policy.path=lerobot/smolvla_base` fails** with "Repo id must use alphanumeric chars…:
  'lerobot\smolvla_base'": the train config stores the path as a `Path`, which uses `\` on
  Windows. `m9_finetune.py` passes the downloaded snapshot directory instead.
- **Checkpoint symlink**: after each checkpoint `lerobot-train` creates a `last` symlink, which
  Windows refuses without Developer Mode or admin rights (`WinError 1314`); training then stops.
  The checkpoint itself is complete. `m9_finetune.py` therefore saves only at the final step by
  default. Turn on Developer Mode (Settings → System → For developers) to save intermediate
  checkpoints.

## How to run

Windows commands below; for long runs on a Linux GPU machine (no workarounds, video datasets,
batch size 64 on 48 GB), see [setup-linux.md](setup-linux.md).

```powershell
uv sync --extra train                                       # vla + lerobot[training] + mink
uv run python scripts/m8_record_demos.py --episodes 50      # ~1 h, ~2.5 GB
uv run python scripts/m9_finetune.py --episodes 10 --steps 2000
uv run python scripts/m7_zero_shot_eval.py --checkpoint outputs/train/<run>/checkpoints/<step>/pretrained_model `
    --instructions "Pick up the red cube and place it in the green box." --seeds 0 1 2 3 4 5 6 7 8 9
```

## Results

### Recording (2026-11-04)

`m8_record_demos.py --episodes 50`: 50 episodes kept from 52 attempts (seeds 1000–1051),
39,372 frames (~26 s each), 2.4 GB, 64 min on the laptop. Two layouts were skipped before
moving: with the box at the far edge of its range (y ≈ 0.14 m) the "retreat" waypoint was
15.5–16.4 mm off in IK (limit 15 mm). Every kept episode ends with the cube in the box.

### Pilot fine-tune: 10 episodes, 2,000 steps

`m9_finetune.py --episodes 10 --steps 2000 --num-workers 4`: 34 min on the RTX 4060 laptop GPU
(batch 8, 3.6 GB VRAM, ~1 s/step; data loading 0.2 s/step with 4 workers vs 2.9 s with 0),
about 2 epochs. Training loss 0.26 (step 100) → 0.035 (step 2,000).

Evaluation (`m7_zero_shot_eval.py --checkpoint … --seeds 0…9 --seconds 40`), unseen layouts,
training instruction, whole 50-action chunks:

| | zero-shot `smolvla_base` (Stage 2, checkpoint stats) | fine-tuned, 10 ep × 2k steps |
|---|---|---|
| L1 inference | 9/9 | 10/10 |
| L2 control (no arm–table contact) | 0/9 | 7/10 |
| touched cube | 5/9 | 7/10 |
| grasped (both jaws) | 0/9 | 1/10 |
| min gripper–cube distance (mean) | 3.8–4.5 cm | 2.6 cm (best 0.5 cm) |
| arm–table contact (ticks of 1200, mean) | ~700–800 (of 900) | 5 |
| **L3 success** | **0/9** | **0/10** |

The fine-tuned policy goes to the cube and closes the gripper there (most "clipped" ticks
are gripper commands slightly below fully closed, clipped to closed), and the arm no longer
presses on the table. It grasped the cube once and never lifted it: with 10 demonstrations
and 2k steps it does not yet grasp reliably. This matches the SmolVLA docs, which recommend
~50 episodes and report 25 as insufficient in their experiments.

Re-planning more often does not help this checkpoint (`--replan-every 10`, same seeds):

| | whole 50-action chunks | re-plan every 10 actions |
|---|---|---|
| L2 control | 7/10 | 9/10 |
| touched / grasped | 7/10 / 1/10 | 3/10 / 0/10 |
| min gripper–cube distance (mean) | 2.6 cm | 4.3 cm |
| L3 success | 0/10 | 0/10 |

With frequent re-planning the arm hovers instead of committing to the approach; whole chunks
(the checkpoint's `n_action_steps` = 50) stay the default.

### 50 episodes, 5,000 steps

`m9_finetune.py --episodes 50 --steps 5000 --num-workers 4`: 74 min, 1.0 epoch over 39,372
frames, loss 0.28 → 0.028. Same evaluation (seeds 0–9, unseen, whole chunks, 40 s):

| | zero-shot | 10 ep × 2k steps | 50 ep × 5k steps |
|---|---|---|---|
| L1 inference | 9/9 | 10/10 | 10/10 |
| L2 control | 0/9 | 7/10 | 9/10 |
| touched cube | 5/9 | 7/10 | 6/10 |
| grasped (both jaws) | 0/9 | 1/10 | 2/10 |
| lifted | 0/9 | 0/10 | **1/10** |
| min gripper–cube distance (mean) | 3.8–4.5 cm | 2.6 cm | 2.8 cm |
| arm–table contact (ticks, mean) | ~750 | 5 | 0.1 |
| **L3 success** | **0/9** | **0/10** | **0/10** |

The best episode (seed 6) shows the whole task structure: the policy grasped the cube at
t ≈ 15 s, lifted it to 8 cm, carried it ~19 cm toward the box and released it, but 7 cm short,
so the cube landed just outside the box wall (not a success; images in the episode folder
under `outputs/ft_ep50_s5000_eval_*`). The other grasp (seed 4) did not lift. In the rest, the
arm reaches and closes near the cube but misses the grasp.

With 10 seeds per checkpoint, the differences between the two fine-tuned rows (1 vs 2 grasps,
0 vs 1 lift) are too small to call a trend; the clear changes are versus zero-shot (no table
contact, reaching the cube).

### 50 episodes, 20,000 steps, batch 16 (overnight, 2026-11-04)

`m9_finetune.py --episodes 50 --steps 20000 --batch-size 16 --num-workers 8`: 317 min on the
laptop (0.9 s/step, 5.84 GB VRAM; the last ~15 min on battery, ~1.1 s/step), ~8 epochs over the
39,372 frames, loss 0.28 → 0.011. No errors besides the expected final 'last'-symlink message.

Evaluation on 20 unseen layouts (seeds 0–19), training instruction, whole chunks, 40 s:

| | zero-shot | 10 ep × 2k | 50 ep × 5k | 50 ep × 20k, run 1 | 50 ep × 20k, run 2 |
|---|---|---|---|---|---|
| seeds | 9 | 10 | 10 | 20 | 20 |
| L1 inference | 9/9 | 10/10 | 10/10 | 20/20 | 20/20 |
| L2 control | 0/9 | 7/10 | 9/10 | 18/20 | 19/20 |
| grasped (both jaws) | 0/9 | 1/10 | 2/10 | 8/20 | 4/20 |
| lifted | 0/9 | 0/10 | 1/10 | 4/20 | 4/20 |
| cube in box, LeRobot test | 0/9 | 0/10 | 0/10 | 4/20 | 4/20 |
| **L3: placed (in box and released)** | **0/9** | **0/10** | **0/10** | ≥1/20 (see below) | **4/20** |

**Verified placements (run 2): seeds 2, 10, 13, 19.** In each, the logged cube ends resting on
the box floor (center z = 1.9 cm), 0.5–1.4 cm from the box center, with no gripper contact, after
20.7–22.5 s (the scripted expert needs ~27 s). Every grasp in run 2 ended in a placement.

**Evaluation bug found and fixed.** Run 1 stopped an episode as successful once LeRobot's
in-box test (cube center inside the box footprint and below 6 cm) had held for 0.5 s. That
test also passes for a cube still held just above the 3.5 cm box walls: in run 1, seeds 2, 13
and 19 stopped with the cube at z ≈ 5.5 cm in the gripper; only seed 17 was resting on the box
floor, so run 1 shows at least 1 placement. `evaluation/rollout.py` now requires the cube to
be in the box *and released* (no gripper contact) for 0.5 s; LeRobot's test is still logged as
`in_box_lerobot_ever`. Run 2 is the same checkpoint and seeds with the fixed rule.

**Run-to-run variation.** The policy samples noise in every flow-matching inference, and the
evaluation does not fix a torch seed, so the same layout can go differently: seed 17 placed
the cube in run 1 but not in run 2, seed 10 the reverse; grasps were 8/20 vs 4/20. With 20
episodes per run, treat "~20% placement" as the estimate, not a precise rate.

Failure modes (run 2, 16 failures): all are failed grasps. In 11 the gripper never touched
the cube (it stopped 4–6 cm away or closed beside it); in 5 (seeds 3, 6, 15, 17, 18) it touched
the cube and pushed it 1.4–6.0 cm without getting both jaws on it; seed 3 also pressed on the
table for 14 ticks (the only Level 2 failure). No failure came from a dropped cube or a missed
box: once grasped, the policy carried and placed it.

### Does it follow language? (language probe)

`m7_zero_shot_eval.py --checkpoint <50 ep × 20k> --seeds 2 10 13 19` (layouts where it had
succeeded) with four instructions; results in `outputs/language_probe_eval_*`:

| Instruction | placed in box | grasped | behaviour |
|---|---|---|---|
| "Pick up the red cube and place it in the green box." (training) | 2/4 | 3/4 | the trained task |
| "Pick up the blue cube and place it in the green box." (no blue cube exists) | 3/4 | 4/4 | picked up the red cube anyway |
| "Do not move." | 1/4 | 4/4 | moved as much as usual (887–2185° of joint travel) and grasped the red cube |
| "Push the red cube to the left." | 0/4 | 1/4 | moved, but never pushed: cube unmoved in 3/4, gripper stayed 3–6 cm away |

The fine-tuned policy ignores the content of the instruction: a non-existent colour and "do not
move" both produce the trained pick-and-place. Unfamiliar wording ("push … left") disrupts the
trained motion without producing the requested one. Expected: every demonstration used the same
sentence, so nothing tied words to behaviour. Language-conditioned behaviour needs demonstrations
with different instructions for different behaviours (e.g. the official scene's four coloured
cubes, "pick up the <colour> cube"). 4 episodes per instruction, so treat the counts as indicative.

### Shaking: diagnosis and Real-Time Chunking

In the viewer the arm visibly shakes. Measured on seed 2 (50 ep × 20k checkpoint):

| | expert demos | policy, whole chunks | policy + RTC |
|---|---|---|---|
| joint action acceleration, median / p95 [rad/s²] | 0.0 / 0.65 | 6.1 / 22.1 | 5.3 / 22.7 |
| jump at a chunk boundary, median / max | – | 1.0° / 5.6° | 0.59° / 2.0° |
| measured joint velocity sign changes per tick | – | 0.41 | 0.41 |
| ticks rate-limited by the safety filter | – | 1160 / 1200 | 1098 / 1200 |
| inference per chunk | – | 0.8 s | 1.3 s (every 20 actions) |

The demonstrations are smooth; the policy's predicted actions are ~30× rougher, so the safety
filter's 6 rad/s² limit trims almost every tick and the measured velocity changes direction
about every other tick (the visible ~15 Hz buzz), plus a twitch at each chunk boundary.

LeRobot's Real-Time Chunking (`--rtc`; [docs](https://huggingface.co/docs/lerobot/rtc), implemented
as in `lerobot/rollout/inference/rtc.py`: re-plan at 30 actions left, guide the new chunk with
the unexecuted tail, `execution_horizon` 10, `max_guidance_weight` 10, EXP schedule,
`inference_delay` 0 since our loop is synchronous) halves the boundary jumps, as documented, but
does not change the jitter inside chunks, which is the main source of the shaking. (RTC needs
gradients inside its guidance step, so the wrapper uses `torch.no_grad` instead of
`inference_mode` when it is on.) LeRobot's `ActionInterpolator` (`--interpolation_multiplier`)
only interpolates between actions at a higher control rate, so it would not remove the jitter
either.

Next: more evaluation seeds (and fixed torch seeds) for a tighter estimate; 100–200
demonstrations (design.md step 7) and more varied grasps to improve the grasp rate; then real
leader/follower demonstrations.
