# Windows setup

Tested on Windows 11 with Python 3.12, uv 0.12, and MuJoCo 3.17. Stage 1 needs no GPU or CUDA.

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
| numpy | 2.5.4 | >=2.0, <3 | |
| pyyaml | 6.0.3 | >=6.0, <7 | |
| pytest (dev) | 9.2.0 | >=8.0 | |

Stage 2 (torch, LeRobot, SmolVLA) goes in the `vla` extra so Stage 1 never pulls in CUDA.

## 3. Check it works

```powershell
uv run pytest                                   # headless tests, about 10 s
uv run python scripts/m1_inspect_model.py       # print joints/actuators
uv run python scripts/m1_view_model.py          # viewer window, arm holding "rest"
uv run python scripts/m2_joint_demo.py          # viewer, joint-by-joint + trajectory demo
uv run python scripts/m2_gripper_demo.py        # gripper open/close
uv run python scripts/m3_teleop_demo.py         # slider window drives the arm (virtual leader)
uv run python scripts/m3_teleop_demo.py --leader sweep --headless --switch-demo
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
- Run scripts from the repo root. Config paths are relative to it.
