"""Fetch LeRobot's official SO-101 tabletop scene into assets/lerobot_tabletop/.

Source: https://huggingface.co/lerobot/so101-tabletop-mujoco (Apache-2.0), the
EnvHub environment LeRobot evaluates SO-101 policies in. We take the scene XML
and its texture only; the arm is our committed Menagerie copy (same commit),
attached by so101_vla.sim.scene. The files are committed, so this script is
only needed to re-fetch them.

Usage:
    uv run python scripts/fetch_tabletop_scene.py [--revision SHA]
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

REPO_ID = "lerobot/so101-tabletop-mujoco"
PINNED_REVISION = "844dee6c058960c2ef42b65e2c68b47a5cf7c4f3"  # 2026-10-28
FILES = ["so101_tabletop.xml", "textures/wood.png", "README.md"]

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "assets" / "lerobot_tabletop"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--revision", default=PINNED_REVISION)
    args = parser.parse_args()

    for name in FILES:
        url = f"https://huggingface.co/{REPO_ID}/resolve/{args.revision}/{name}"
        out = DEST / name
        out.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as r:  # noqa: S310 - fixed https URL
            out.write_bytes(r.read())
        print(f"{name}: {out.stat().st_size} bytes")

    (DEST / "PROVENANCE.md").write_text(
        "# Provenance\n\n"
        f"Copied from https://huggingface.co/{REPO_ID} at revision `{args.revision}`.\n\n"
        "License: Apache-2.0, including `textures/wood.png` (see README.md, \"Licenses\").\n"
        "`so101_tabletop.xml`, `textures/wood.png` and `README.md` are unmodified.\n"
        "`so101_pick_place.xml` is ours, derived from `so101_tabletop.xml` (one cube, see its header).\n"
        "Re-fetch with `uv run python scripts/fetch_tabletop_scene.py`.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
