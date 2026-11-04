"""A VLA policy as a teleop-style command source (Task 2.6, and Task 1.6's "VLA policy" source).

``PolicySource.get_action()`` has the same contract as a LeRobot teleoperator, so the
policy plugs into ``CommandSelector`` next to the leaders: one active source, safe
engage on switch, then the Stage 1 safety filter. Chunk execution, as in the
SmolVLA paper's real-robot setup and ``SmolVLAPolicy.select_action``: predict a
chunk, play it one action per control tick, re-observe and predict again when it
runs out (or after ``replan_every`` actions). Physics (210 Hz), control (30 Hz) and
inference (once per chunk) run at separate rates.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from so101_vla.conventions import LEROBOT_FEATURE_NAMES, LeRobotUnits
from so101_vla.policies.action_adapter import ActionChunkValidator, ChunkReport


class ChunkPolicy(Protocol):
    camera_names: list[str]
    last_inference_ms: float | None

    def reset(self) -> None: ...
    def predict_chunk(self, observation: dict, instruction: str) -> np.ndarray: ...


@dataclass
class ChunkRecord:
    time: float  # sim time of the observation the chunk was predicted from
    instruction: str
    chunk: np.ndarray  # (H, 6) raw policy output, LeRobot units
    report: ChunkReport
    inference_ms: float | None
    frames: dict[str, np.ndarray] = field(default_factory=dict)  # observation images used


class PolicySource:
    def __init__(self, policy: ChunkPolicy, observe: Callable[[], dict], clock: Callable[[], float],
                 current_q: Callable[[], np.ndarray], validator: ActionChunkValidator, instruction: str,
                 replan_every: int | None = None, keep_frames: bool = True) -> None:
        """
        observe:       returns ``RobotInterface.get_observation(cameras=policy.camera_names, ...)``
        current_q:     measured joint positions [rad] (for the validation report)
        replan_every:  predict a new chunk after this many actions (None = play whole chunks)
        """
        self.policy, self.observe, self.clock, self.current_q = policy, observe, clock, current_q
        self.validator, self.instruction = validator, instruction
        self.replan_every, self.keep_frames = replan_every, keep_frames
        self.chunks: list[ChunkRecord] = []
        self.executed: list[tuple[float, np.ndarray]] = []  # (sim time, action in LeRobot units)
        self._queue: deque[np.ndarray] = deque()
        self._played = 0

    @classmethod
    def for_robot(cls, robot, policy: ChunkPolicy, instruction: str, arm: str = "main",
                  replan_every: int | None = None, keep_frames: bool = True,
                  image_size: tuple[int, int] | None = None) -> PolicySource:
        """Standard wiring for one arm of a ``RobotInterface``: observation, clock, state, validator."""
        w, h = image_size or getattr(getattr(policy, "config", None), "image_size", (256, 256))
        lim = robot.joint_limits(arm)
        return cls(policy, observe=lambda: robot.get_observation(cameras=policy.camera_names, width=w, height=h),
                   clock=lambda: robot.get_state().time, current_q=lambda: robot.get_state().arms[arm].positions,
                   validator=ActionChunkValidator(LeRobotUnits(lim.lower, lim.upper)), instruction=instruction,
                   replan_every=replan_every, keep_frames=keep_frames)

    def reset(self, instruction: str | None = None) -> None:
        if instruction is not None:
            self.instruction = instruction
        self._queue.clear()
        self._played = 0
        self.policy.reset()

    def _replan(self) -> None:
        t = self.clock()
        obs = self.observe()
        chunk = np.asarray(self.policy.predict_chunk(obs, self.instruction), dtype=np.float64)
        report = self.validator.validate(chunk, self.current_q())  # raises InvalidChunk -> selector holds
        frames = {c: obs[c] for c in self.policy.camera_names} if self.keep_frames else {}
        self.chunks.append(ChunkRecord(t, self.instruction, chunk, report, self.policy.last_inference_ms, frames))
        self._queue = deque(chunk)
        self._played = 0

    def get_action(self) -> dict[str, float]:
        if not self._queue or (self.replan_every is not None and self._played >= self.replan_every):
            self._replan()
        action = self._queue.popleft()
        self._played += 1
        self.executed.append((self.clock(), action))
        return dict(zip(LEROBOT_FEATURE_NAMES, action.tolist(), strict=True))

    # --- logs --------------------------------------------------------------------------

    def arrays(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {
            "chunk_time": np.array([c.time for c in self.chunks]),
            "chunks": np.array([c.chunk for c in self.chunks]) if self.chunks else np.zeros((0, 0, 6)),
            "inference_ms": np.array([np.nan if c.inference_ms is None else c.inference_ms for c in self.chunks]),
            "executed_time": np.array([t for t, _ in self.executed]),
            "executed_lerobot": np.array([a for _, a in self.executed]) if self.executed else np.zeros((0, 6)),
        }
        for cam in self.policy.camera_names if self.keep_frames and self.chunks else ():
            out[f"frames/{cam.replace('/', '_')}"] = np.array([c.frames[cam] for c in self.chunks])
        return out
