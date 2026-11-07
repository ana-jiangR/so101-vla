"""Milestone 9 (Stage 3): fine-tune smolvla_base on the recorded demonstrations with lerobot-train.

This only assembles the official command from the SmolVLA docs
(huggingface.co/docs/lerobot/smolvla, "Finetune SmolVLA on your data"):

    lerobot-train --policy.path=lerobot/smolvla_base --dataset.repo_id=... --batch_size=...
                  --steps=... --output_dir=... --policy.device=cuda

plus what this setup needs:

* ``--dataset.root``: the local dataset from m8_record_demos.py (no Hub upload)
* ``--rename_map``: dataset cameras top/wrist/front -> the checkpoint's camera1/2/3
  (the SmolVLA paper's top, wrist, side order; lerobot-train supports rename_map
  for pretrained policies)
* ``--dataset.episodes``: train on the first N episodes (data-efficiency runs)
* Windows workarounds, both in LeRobot 0.6.1:
  - ``--policy.path`` is given as the local snapshot directory: the config stores it
    as a ``Path``, which turns "lerobot/smolvla_base" into "lerobot\\smolvla_base".
  - checkpoints end with a ``last`` symlink, which Windows only allows with Developer
    Mode (or admin). Without it, training stops right after its first checkpoint, so
    by default we save once, at the final step. Enable Developer Mode to save more often.

    uv run python scripts/m9_finetune.py --episodes 10 --steps 2000            # pilot
    uv run python scripts/m9_finetune.py --episodes 50 --steps 20000           # docs' recommendation
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

from so101_vla.config import PROJECT_ROOT, resolve_path

BASE = "lerobot/smolvla_base"
BASE_REVISION = "5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6"
RENAME = {"observation.images.top": "observation.images.camera1",
          "observation.images.wrist": "observation.images.camera2",
          "observation.images.front": "observation.images.camera3"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset-root", default="outputs/datasets/so101_pick_place_sim")
    ap.add_argument("--repo-id", default="local/so101_pick_place_sim")
    ap.add_argument("--episodes", type=int, default=None, help="use the first N episodes (default: all)")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=8, help="8 fits in ~3.6 GB on an 8 GB laptop GPU")
    ap.add_argument("--save-freq", type=int, default=None, help="default: only at the end (see Windows note)")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--dry-run", action="store_true", help="print the command only")
    ap.add_argument("--stock-query", action="store_true",
                    help="use LeRobot's own action-chunk query (decodes every image of the chunk; ~15x slower "
                         "data loading on image datasets, see so101_vla/lerobot_train_fast.py)")
    args, extra = ap.parse_known_args()  # anything else is passed to lerobot-train unchanged

    from huggingface_hub import snapshot_download

    base_dir = snapshot_download(BASE, revision=BASE_REVISION)
    tag = f"ep{args.episodes or 'all'}_steps{args.steps}"
    out = resolve_path(args.output_dir or f"outputs/train/smolvla_so101_pick_place_{tag}")
    module = "lerobot.scripts.lerobot_train" if args.stock_query else "so101_vla.lerobot_train_fast"
    cmd = [sys.executable, "-m", module,
           f"--policy.path={base_dir}", f"--dataset.repo_id={args.repo_id}",
           f"--dataset.root={resolve_path(args.dataset_root)}", f"--rename_map={json.dumps(RENAME)}",
           f"--batch_size={args.batch_size}", f"--steps={args.steps}",
           f"--save_freq={args.save_freq or args.steps}", f"--log_freq={max(1, min(100, args.steps // 20))}",
           f"--num_workers={args.num_workers}", f"--seed={args.seed}", f"--output_dir={out}",
           f"--job_name=smolvla_so101_{tag}", "--policy.device=cuda", "--policy.push_to_hub=false",
           "--wandb.enable=false"]
    if args.episodes:
        cmd.append(f"--dataset.episodes={json.dumps(list(range(args.episodes)))}")
    cmd += extra
    print(" ".join(cmd))
    if args.dry_run:
        return
    t0 = time.perf_counter()
    code = subprocess.call(cmd, cwd=PROJECT_ROOT)
    final = out / "checkpoints" / f"{args.steps:06d}" / "pretrained_model"
    if (final / "model.safetensors").is_file():
        note = "" if code == 0 else " (lerobot-train exited after saving; on Windows this is the 'last' symlink)"
        print(f"\nfine-tuned checkpoint: {final}{note}  [{(time.perf_counter() - t0) / 60:.1f} min]")
        print(f"evaluate: uv run python scripts/m7_zero_shot_eval.py --checkpoint \"{final}\"")
    else:
        raise SystemExit(f"training failed (exit {code}); no checkpoint at {final}")


if __name__ == "__main__":
    main()
