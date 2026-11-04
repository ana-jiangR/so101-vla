"""Fetch the MuJoCo Menagerie SO-101 model into assets/robotstudio_so101/.

We use the same model and commit as LeRobot's official MuJoCo environment
(huggingface.co/lerobot/so101-tabletop-mujoco), so our simulation matches the
one LeRobot policies are evaluated in. Menagerie's model is derived from
TheRobotStudio's so101_new_calib.xml, with gripper collision primitives,
implicitfast integration, and a wrist camera added (see its README/CHANGELOG).

The copied files are committed to this repo, so you only need to re-run this
script when deliberately upgrading the model.

Usage:
    uv run python scripts/fetch_so101_assets.py [--commit SHA]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"
# Commit pinned by lerobot/so101-tabletop-mujoco (its README / env.py).
PINNED_COMMIT = "0059d4335f8156206f63a35662313385f7ad6d74"
MODEL_DIR = "robotstudio_so101"

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "assets" / MODEL_DIR

# Everything we need from the model directory. scene*.xml and the preview
# image are skipped: our own scenes live in assets/scenes/.
FILES = ["so101.xml", "README.md", "CHANGELOG.md", "LICENSE"]


def run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--commit", default=PINNED_COMMIT)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "menagerie"
        run(["git", "init", "-q", str(repo)])
        run(["git", "remote", "add", "origin", REPO_URL], cwd=repo)
        run(["git", "sparse-checkout", "set", "--no-cone", f"/{MODEL_DIR}/"], cwd=repo)
        run(["git", "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", args.commit], cwd=repo)
        run(["git", "checkout", "-q", "FETCH_HEAD"], cwd=repo)

        src = repo / MODEL_DIR
        if DEST.exists():
            shutil.rmtree(DEST)
        DEST.mkdir(parents=True)
        for name in FILES:
            shutil.copy2(src / name, DEST / name)
        shutil.copytree(src / "assets", DEST / "assets")
        n_assets = sum(1 for _ in (DEST / "assets").iterdir())

    (DEST / "PROVENANCE.md").write_text(
        "# Provenance\n\n"
        f"Copied from {REPO_URL.removesuffix('.git')}/tree/{args.commit}/{MODEL_DIR}\n"
        "(the commit pinned by https://huggingface.co/lerobot/so101-tabletop-mujoco).\n\n"
        "License: Apache-2.0 (see LICENSE). Files are unmodified.\n"
        "Regenerate with `uv run python scripts/fetch_so101_assets.py`.\n",
        encoding="utf-8",
    )
    print(f"Copied {len(FILES)} files and {n_assets} assets to {DEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
