# Windows setup

Tested on Windows 11 with Python 3.12, uv 0.12, and MuJoCo 3.17. Stage 1 needs no GPU or CUDA.
For long fine-tuning runs on a Linux GPU machine (or WSL2), see [setup-linux.md](setup-linux.md).

## 1. Install uv

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart the terminal so `uv` is on your PATH.

## 2. Create the environment

From the repo root:

```powershell
uv sync
```

This command:

- installs Python 3.12 if needed (pinned in `.python-version`)
- creates `.venv\`
- installs the exact versions in `uv.lock`, including this package in editable mode

### Versions

`uv.lock` pins every package exactly; `pyproject.toml` gives the compatible ranges.

| Package | Tested | Allowed | Why the bounds |
|---|---|---|---|
| Python | 3.12.14 | 3.12.x | design brief; LeRobot supports 3.12 |
| mujoco | 3.17.0 | >=3.3, <4 | needs `MjSpec.attach` and `mj_step(nstep=)`; 4.x may change the API |
| numpy | 2.2.6 | >=2.0, <3 | lerobot 0.6.1 needs <2.3, and uv resolves one version for all extras |
| pyyaml | 6.0.3 | >=6.0, <7 | |
| pytest (dev) | 9.2.0 | >=8.0 | |
| lerobot[smolvla] (`vla` extra) | 0.6.1 | ==0.6.1 | Stage 2 calls its APIs directly; pinned so they don't move |
| torch (`vla` extra) | 2.11.0+cu128 | >=2.7, <2.12 | lerobot 0.6.1's range; CUDA 12.8 wheels from download.pytorch.org |
| transformers (`vla` extra) | 5.5.4 | via lerobot | |
| mink (`train` extra) | 1.4.0 | >=1.4, <2 | differential IK for the scripted expert |

Stage 2 (torch, LeRobot, SmolVLA) is in the `vla` extra, so a plain `uv sync` never pulls in CUDA.

## 2b. Stage 2 (SmolVLA): GPU environment

> **Note:** `uv sync` makes the environment match *exactly* the extras you name, so a plain `uv sync`
> uninstalls torch/LeRobot/mink if they were installed (then `m5`–`m9` fail with
> `No module named 'torch'`). Once you use Stage 2/3, always run `uv sync --extra train`.
> `uv run ...` by itself does not remove packages.


```powershell
uv sync --extra vla
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

- About 3 GB of wheels (PyTorch with CUDA 12.8). Tested on an RTX 4060 Laptop GPU (8 GB) with
  driver 617; SmolVLA needs about 0.9 GiB of VRAM for inference.
- The first `m5_offline_inference.py` run downloads `lerobot/smolvla_base` (~1 GB) and the SmolVLM2
  tokenizer into `%USERPROFILE%\.cache\huggingface`. No Hugging Face login is needed.
- Hugging Face prints "To support symlinks on Windows, you either need to activate Developer Mode…".
  It is harmless (the cache copies files instead). Set `HF_HUB_DISABLE_SYMLINKS_WARNING=1` to hide it.
- No GPU: pass `--device cpu` to `m5_offline_inference.py` (much slower).

## 2c. Stage 3 (demonstrations + fine-tuning)

```powershell
uv sync --extra train      # adds lerobot[training] (datasets, PyAV, torchcodec, wandb) and mink
```

- Recording needs no GPU; fine-tuning uses ~3.6 GB VRAM at batch size 8.
- torchcodec prints a long "Could not load libtorchcodec" traceback on import. LeRobot then falls
  back to PyAV, and our image (non-video) datasets do not need it. To use video datasets, install a
  shared FFmpeg build (e.g. `winget install Gyan.FFmpeg.Shared`) and record with `use_videos=True`.
- `lerobot-train` creates a `last` symlink after each checkpoint; Windows needs Developer Mode for
  that (Settings → System → For developers). Without it `scripts/m9_finetune.py` saves only at the end.
- See [stage3.md](stage3.md).

## 3. Check it works

```powershell
uv run pytest                                   # headless tests, about 10 s
uv run python scripts/m1_inspect_model.py       # print joints/actuators
uv run python scripts/m1_view_model.py          # viewer window, arm holding "rest"
uv run python scripts/m2_joint_demo.py          # viewer, joint-by-joint + trajectory demo
uv run python scripts/m2_gripper_demo.py        # gripper open/close
uv run python scripts/m3_teleop_demo.py         # slider window drives the arm (virtual leader)
uv run python scripts/m3_teleop_demo.py --leader sweep --headless --switch-demo

# Stage 2 (after `uv sync --extra vla`)
uv run python scripts/m4_tabletop_scene.py      # tabletop scene, saves every camera view
uv run python scripts/m5_offline_inference.py   # SmolVLA inference only, robot does not move
uv run python scripts/m6_vla_closed_loop.py     # SmolVLA drives the arm, viewer in real time
uv run python scripts/m7_zero_shot_eval.py      # zero-shot evaluation, headless (~10 min)

# Stage 3 (after `uv sync --extra train`)
uv run python scripts/m8_record_demos.py --episodes 10   # scripted demonstrations -> LeRobotDataset
uv run python scripts/m9_finetune.py --episodes 10 --steps 2000
```

Add `--headless` to the m2 demos (and to `m3_teleop_demo.py --leader sweep`) to run
without a window, e.g. over SSH. Each demo saves its logs under `outputs/`.

Note: run Python through `uv run`. On this machine a bare `python` resolves to
the Windows Store stub, not the project environment.

## Notes

- The robot assets in `assets/robotstudio_so101/` (MuJoCo Menagerie, commit
  `0059d43`, the one LeRobot's SO-101 env pins) are committed. Re-run
  `uv run python scripts/fetch_so101_assets.py` only to re-fetch them (this needs `git`).
- The viewer uses OpenGL through GLFW. If the window fails to open on a laptop
  with two GPUs, update your graphics drivers, or choose the NVIDIA GPU for
  `python.exe` in Windows Graphics settings.
- The tabletop scene in `assets/lerobot_tabletop/` is LeRobot's official one (revision `844dee6`),
  plus our one-cube variant. `uv run python scripts/fetch_tabletop_scene.py` re-fetches it.
- Run scripts from the repo root. Config paths are relative to it.
