"""Milestone 12: record real demonstrations with LeRobot's ``lerobot-record`` (quiet, arrow keys only).

Runs the official recorder unchanged, with three usability changes for Windows:

1. Keys: ``lerobot-record`` listens to the keyboard *globally* (pynput, every window) and also
   treats the letters n / r / q as next / re-record / quit, so typing anywhere ends or discards
   episodes. Here only Right / Left / Esc count, once per physical press (Windows auto-repeat on
   a held key would otherwise end the episode *and* skip the reset period).
2. Quiet output: hides the torchcodec/FFmpeg traceback printed at import (harmless: LeRobot falls
   back to PyAV), the configuration dump, and the dataset-saving progress bars.
3. Clear status banners: RECORDING / RESET / SAVING / DISCARDED, plus all real warnings and errors.

All arguments are passed to ``lerobot-record`` as-is, e.g.:

    uv run python scripts/m12_record_real.py --robot.type=so101_follower --robot.port=COM12 ...

Controls: Right = episode done (next) / end reset early, Left = discard and re-record,
Esc = stop and save. Arrow keys are still global: avoid them in other windows while recording.
"""

from __future__ import annotations

import contextlib
import io
import logging
import time

# Import LeRobot with stderr captured: torchcodec prints a long "Could not load libtorchcodec"
# traceback here when FFmpeg is not installed (LeRobot then uses PyAV; images need neither).
_import_noise = io.StringIO()
with contextlib.redirect_stderr(_import_noise):
    import datasets
    import lerobot.scripts.lerobot_record as lerobot_record
    import lerobot.utils.keyboard_input as keyboard_input
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

# --- 1. keys: Right / Left / Esc only, one event per press -------------------------------------

ALLOWED_KEYS = {"right", "left", "esc"}
REPEAT_GAP_S = 0.6  # Windows auto-repeat sends a press every ~30 ms while a key is held
_resolve = keyboard_input._resolve_pynput_key
_last_seen: dict[str, float] = {}


def _arrows_and_esc_only(key):
    name = _resolve(key)
    if name not in ALLOWED_KEYS:
        return None
    now = time.monotonic()
    previous = _last_seen.get(name)
    _last_seen[name] = now
    if previous is not None and now - previous < REPEAT_GAP_S:
        return None  # auto-repeat of a held key, or an accidental double tap
    return name


keyboard_input._resolve_pynput_key = _arrows_and_esc_only  # used by create_key_listener's on_press

# --- 2. quiet logs ----------------------------------------------------------------------------------


class _QuietFilter(logging.Filter):
    """Keep warnings/errors (except the torchcodec notice) and 'connected' lines; drop other INFO."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "torchcodec" in msg or "libtorchcodec" in msg or "Streaming encoding is disabled" in msg:
            return False
        if "Relative goal position magnitude had to be clamped" in msg:
            return False  # expected with --robot.max_relative_target: the follower catching up, every tick
        if record.levelno >= logging.WARNING:
            return True
        return "connected" in msg and len(msg) < 200


_init_logging = lerobot_record.init_logging


def _quiet_init_logging(*args, **kwargs):
    _init_logging(*args, **kwargs)
    for handler in logging.getLogger().handlers:
        handler.addFilter(_QuietFilter())


lerobot_record.init_logging = _quiet_init_logging
datasets.disable_progress_bars()  # the "Map: ..." bars shown while saving an episode

# --- 3. status banners ------------------------------------------------------------------------------

_reset_s = None


def _banner(text: str, play_sounds: bool = True, blocking: bool = False) -> None:
    clock = time.strftime("%H:%M:%S")
    if text.startswith("Recording episode"):
        n = text.split()[-1]
        print(f"\n[{clock}] >>> RECORDING episode {n}: pick up the cube with the leader, then tap RIGHT once")
    elif text.startswith("Reset the environment"):
        span = f"{_reset_s:g} s" if _reset_s else "reset time"
        print(f"[{clock}] ... RESET ({span}, leader still controls the follower): put the cube down at a new "
              "spot, open the gripper, move back to the start pose. Tap RIGHT to finish early.")
    elif text.startswith("Re-record episode"):
        print(f"[{clock}] xxx DISCARDED: this episode will be recorded again")
    elif text.startswith("Stop recording"):
        print(f"[{clock}] === STOPPING: saving and disconnecting ...")
    elif text.startswith("Exiting"):
        print(f"[{clock}] === DONE")
    else:
        print(f"[{clock}] {text}")


lerobot_record.log_say = _banner

_save_episode = LeRobotDataset.save_episode


def _save_episode_with_status(self, *args, **kwargs):
    frames = self.writer.episode_buffer["size"] if getattr(self, "writer", None) and self.writer.episode_buffer else "?"
    t0 = time.perf_counter()
    print(f"[{time.strftime('%H:%M:%S')}] ... SAVING episode {self.num_episodes} ({frames} frames): "
          "follower frozen, don't touch it ...", flush=True)
    result = _save_episode(self, *args, **kwargs)
    print(f"[{time.strftime('%H:%M:%S')}] ... saved ({time.perf_counter() - t0:.0f} s); total episodes: "
          f"{self.num_episodes}", flush=True)
    return result


LeRobotDataset.save_episode = _save_episode_with_status

# --- 4. ignore key presses made while LeRobot was busy (saving) -------------------------------------
# A Right press during saving sets events["exit_early"], which nothing clears until the *next* loop
# starts, so that loop (the next episode) ends instantly with 0 frames and saving it then crashes.
# Clear stale flags when a loop starts, and skip saving an empty episode.
_record_loop = lerobot_record.record_loop


def _record_loop_fresh(*args, **kwargs):
    events = kwargs.get("events", args[1] if len(args) > 1 else None)
    if events is not None and events.get("exit_early") and not events.get("stop_recording"):
        events["exit_early"] = False
    return _record_loop(*args, **kwargs)


lerobot_record.record_loop = _record_loop_fresh
_save_with_status = LeRobotDataset.save_episode


def _save_unless_empty(self, *args, **kwargs):
    if getattr(self, "writer", None) is not None and not self.has_pending_frames():
        print(f"[{time.strftime('%H:%M:%S')}] ... empty episode skipped (nothing recorded)", flush=True)
        return None
    return _save_with_status(self, *args, **kwargs)


LeRobotDataset.save_episode = _save_unless_empty


def main() -> None:
    global _reset_s
    import sys

    for arg in sys.argv[1:]:
        if arg.startswith("--dataset.reset_time_s="):
            _reset_s = float(arg.split("=", 1)[1])
    print("[m12] keys: RIGHT = episode done / end reset, LEFT = discard and redo, ESC = stop and save "
          "(letters and held keys ignored). Connecting ...")
    lerobot_record.main()


if __name__ == "__main__":
    main()
