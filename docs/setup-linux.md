# Linux setup (GPU workstation or WSL2)

Use Linux for long fine-tuning runs (Stage 3): LeRobot is developed on Linux, so the Windows
workarounds in [setup-windows.md](setup-windows.md) are not needed, and a large-memory GPU
(e.g. RTX A6000 / 6000 Ada, 48 GB) allows the batch sizes from the SmolVLA docs.

> Status: the project is developed and tested on Windows 11. The steps below follow the
> official uv, PyTorch, LeRobot and MuJoCo instructions, but have **not been run on Linux in
> this repo yet**. Please report anything that differs.

## What changes compared with Windows

| | Windows (laptop) | Linux |
|---|---|---|
| `--policy.path=lerobot/smolvla_base` | breaks (`lerobot\smolvla_base`); `m9_finetune.py` passes the local snapshot instead | works; the same workaround is harmless |
| Checkpoint `last` symlink | needs Developer Mode, so by default we save only at the end | works: pass `--save-freq` for intermediate checkpoints and resuming |
| Video datasets (torchcodec) | need a shared FFmpeg build; we store images | install FFmpeg, record with `--videos` (much smaller datasets) |
| Headless rendering (cameras, recording, evaluation) | uses the desktop's OpenGL | on a server with no display set `MUJOCO_GL=egl` |
| LeRobot's own IK (`placo`) | no wheels | installable, but not needed (`mink` is used on both) |

## 1. Prerequisites

- An NVIDIA driver recent enough for CUDA 12.8 wheels (the project installs PyTorch
  2.11.0+cu128 from download.pytorch.org). Check with `nvidia-smi` ("CUDA Version" ≥ 12.8).
- `git`, and for video datasets FFmpeg (torchcodec supports FFmpeg 4–8):

```bash
sudo apt-get update && sudo apt-get install -y git ffmpeg
# headless servers: EGL for offscreen MuJoCo rendering (usually provided by the NVIDIA driver)
sudo apt-get install -y libegl1
```

- uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 2. Get the code and install

```bash
git clone https://github.com/ana-jiangR/sol101-vla.git so101-vla
cd so101-vla
uv sync --extra train          # Stage 1 + SmolVLA (CUDA PyTorch) + lerobot[training] + mink
uv run pytest -q               # headless tests; GPU tests run if the model is downloaded
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

On a machine with no display, export this before anything that renders cameras (tests,
`m4`–`m8`, evaluation):

```bash
export MUJOCO_GL=egl
```

The interactive viewer (`--viewer`, `m1_view_model.py`, `m3_teleop_demo.py` sliders) needs a
desktop session; use the `--headless` options on a server.

## 3. Train on Linux

Either record the dataset on the Linux machine (no GPU needed, ~1 h for 50 episodes):

```bash
uv run python scripts/m8_record_demos.py --episodes 50 --videos
```

or copy one recorded on Windows (image datasets work on Linux too):

```bash
# from the Windows machine (PowerShell or Git Bash), e.g.
scp -r outputs/datasets/so101_pick_place_sim user@workstation:~/so101-vla/outputs/datasets/
```

Then fine-tune. On a 48 GB GPU the SmolVLA docs' settings fit:

```bash
uv run python scripts/m9_finetune.py --episodes 50 --steps 20000 --batch-size 64 \
    --save-freq 5000 --num-workers 8
```

`m9_finetune.py` only builds the official `lerobot-train` command (it prints it first); add
`--dry-run` to see it, and any extra `--flag=value` is passed through to `lerobot-train`. To
run in the background on a server:

```bash
nohup uv run python scripts/m9_finetune.py --episodes 50 --steps 20000 --batch-size 64 \
    --save-freq 5000 > outputs/logs/train.log 2>&1 &
tail -f outputs/logs/train.log
```

## 4. Evaluate

On the same machine (`MUJOCO_GL=egl` if headless):

```bash
uv run python scripts/m7_zero_shot_eval.py \
    --checkpoint outputs/train/<run>/checkpoints/020000/pretrained_model \
    --instructions "Pick up the red cube and place it in the green box." --seeds 0 1 2 3 4 5 6 7 8 9
```

or copy the `pretrained_model` folder (~1.3 GB) back to the laptop and run the same command
there, e.g. with the viewer: `uv run python scripts/m6_vla_closed_loop.py --checkpoint <dir>`.

## 5. WSL2 instead of a separate machine

On the Windows laptop, WSL2 (Ubuntu) gives the Linux behaviour above with the same GPU:
install the regular Windows NVIDIA driver (WSL2 uses it; do not install a Linux driver inside
WSL), then follow sections 1–4 inside Ubuntu. Keep the repo on the Linux filesystem
(`~/so101-vla`), not under `/mnt/f/...`, for fast file access during training.
