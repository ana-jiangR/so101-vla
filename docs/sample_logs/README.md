# Sample logs (Stage 1)

These logs were produced headless on 2026-11-03 (MuJoCo 3.17, 30 Hz control). They are
committed so you can see the output format without running anything. New runs go to
`outputs/`, which is gitignored.

| Folder | Command | Contents |
|---|---|---|
| `joint_demo/` | `uv run python scripts/m2_joint_demo.py --headless` | each joint ±0.4 rad, gripper, waypoint trajectory |
| `teleop_sweep_switch/` | `uv run python scripts/m3_teleop_demo.py --leader sweep --headless --switch-demo` | virtual leader → policy stand-in → leader, 20 s |

Files:

- `tracking.npz` / `tracking_summary.json`: from `ArmController` (every tick). Arrays `time`,
  `commanded`, `measured`, `clipped`, `rate_limited`, `relative_limited` (rad). The summary has
  per-joint max/RMS tracking error and how many ticks the safety filter changed.
- `teleop.npz` / `teleop_summary.json`: from `CommandSelector`. Arrays `time` (sample time),
  `source`, `engaged`, `leader_lerobot` (raw LeRobot units), `leader`, `commanded`, `measured` (rad),
  `compute_ms`. The summary has the leader→follower error, the follower lag, the compute latency,
  and the source switch events.

In `teleop_sweep_switch`, all joints lag by 33.3 ms (one control period), the RMS
leader→follower error is at most 0.007 rad, and compute takes 0.42 ms per tick on average
against a 33 ms budget.

```python
import numpy as np
log = np.load("docs/sample_logs/teleop_sweep_switch/teleop.npz")
print(log["joint_names"], log["leader"].shape)   # (600, 6)
```

Replay one of these logs on the simulated follower:

```powershell
uv run python scripts/m3_teleop_demo.py --leader replay --log docs/sample_logs/teleop_sweep_switch/teleop.npz
```

## Stage 2 (SmolVLA, zero-shot)

| Folder | Command | Contents |
|---|---|---|
| `stage2_offline_inference_as_shipped/`, `…_checkpoint_action_stats/` | `uv run python scripts/m5_offline_inference.py [--stats-mode …]` | the three 256×256 input images, the state, three predicted 50×6 chunks (`sample.npz`), the checkpoint description and the validation report (`sample.json`). The robot is not moved |
| `stage2_zero_shot_eval/` | `uv run python scripts/m7_zero_shot_eval.py` | `results.md` / `results.json` for all 18 episodes, plus one full episode per stats mode ("Place the cube in the target area.", seed 2): `episode.json` (levels, progress, per-chunk validation), the first and last image the policy saw per camera, and the teleop/tracking summaries |

Episode folders under `outputs/` also hold `policy.npz` (every predicted chunk, every executed
action, the camera frames per chunk) and `task_state.npz` (cube and gripper positions and
contact flags every tick); those are left out here for size. Results are discussed in
[docs/stage2.md](../stage2.md).
