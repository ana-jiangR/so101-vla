"""Record simulated demonstrations as a LeRobotDataset (Stage 3, design.md steps 2-3).

Follows ``lerobot-record`` (lerobot/scripts/lerobot_record.py) as closely as a
simulated follower allows:

* features from ``hw_to_dataset_features`` over the same observation/action dicts a
  real ``SO101Follower`` exposes: ``"<joint>.pos"`` floats (``so101_follower`` degrees,
  gripper 0..100) and one ``(H, W, 3)`` image per camera
* each control tick: read the observation -> get the command source's action ->
  send it -> ``build_dataset_frame`` + ``add_frame`` with the task string
* ``save_episode`` per episode, ``finalize`` at the end

The recorded ``action`` is the target the follower actually received in this tick
(after safe engage and the safety filter), converted back to LeRobot units; for a
source that stays within the limits it equals the source's own action.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from so101_vla.conventions import LEROBOT_FEATURE_NAMES
from so101_vla.robot.interface import RobotInterface
from so101_vla.teleop.selector import CommandSelector

# dataset camera name -> MuJoCo camera (SmolVLA pretraining order: top, wrist, side)
DEFAULT_CAMERAS: dict[str, str] = {"top": "top", "wrist": "main/wrist_cam", "front": "front"}


@dataclass(frozen=True)
class RecordingConfig:
    repo_id: str = "local/so101_pick_place_sim"
    root: Path = Path("outputs/datasets/so101_pick_place_sim")
    fps: int = 30
    cameras: tuple[tuple[str, str], ...] = tuple(DEFAULT_CAMERAS.items())
    image_size: tuple[int, int] = (256, 256)  # (width, height), smolvla_base's input size
    use_videos: bool = False  # True needs torchcodec + a shared FFmpeg build on Windows
    robot_type: str = "so101_follower"


def dataset_features(cfg: RecordingConfig) -> dict:
    from lerobot.utils.feature_utils import hw_to_dataset_features

    w, h = cfg.image_size
    obs = {k: float for k in LEROBOT_FEATURE_NAMES} | {name: (h, w, 3) for name, _ in cfg.cameras}
    act = {k: float for k in LEROBOT_FEATURE_NAMES}
    return {**hw_to_dataset_features(obs, "observation", cfg.use_videos),
            **hw_to_dataset_features(act, "action", cfg.use_videos)}


class DemoRecorder:
    def __init__(self, cfg: RecordingConfig) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.cfg = cfg
        self.dataset = LeRobotDataset.create(repo_id=cfg.repo_id, fps=cfg.fps, features=dataset_features(cfg),
                                             root=cfg.root, robot_type=cfg.robot_type, use_videos=cfg.use_videos)
        self.frames_in_episode = 0

    def observation(self, robot: RobotInterface) -> dict:
        """The robot's LeRobot-format observation with cameras renamed to dataset camera names."""
        w, h = self.cfg.image_size
        raw = robot.get_observation(cameras=[cam for _, cam in self.cfg.cameras], width=w, height=h)
        obs = {k: raw[k] for k in LEROBOT_FEATURE_NAMES}
        obs.update({name: raw[cam] for name, cam in self.cfg.cameras})
        return obs

    def step(self, robot: RobotInterface, selector: CommandSelector, task: str) -> None:
        """One control tick, recorded: observe -> act -> add_frame (lerobot-record order)."""
        from lerobot.utils.feature_utils import build_dataset_frame

        obs = self.observation(robot)
        selector.tick()
        sent = selector.to_lerobot(np.asarray(selector.log.commanded[-1]))
        frame = {**build_dataset_frame(self.dataset.features, obs, prefix="observation"),
                 **build_dataset_frame(self.dataset.features, dict(zip(LEROBOT_FEATURE_NAMES, sent.tolist(),
                                                                         strict=True)), prefix="action"),
                 "task": task}
        self.dataset.add_frame(frame)
        self.frames_in_episode += 1

    def save_episode(self) -> None:
        self.dataset.save_episode()
        self.frames_in_episode = 0

    def discard_episode(self) -> None:
        self.dataset.clear_episode_buffer()
        self.frames_in_episode = 0

    def finalize(self) -> None:
        self.dataset.finalize()
