"""Command sources for leader-follower teleoperation.

The teleop loop only needs LeRobot's ``Teleoperator.get_action()``::

    get_action() -> {"shoulder_pan.pos": deg, ..., "gripper.pos": 0..100}

i.e. LeRobot feature names and units (``so101-joint-degrees-v1``). A real
LeRobot leader arm therefore plugs in unchanged::

    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
    leader = SO101Leader(SO101LeaderConfig(port="COM5", id="my_leader", use_degrees=True))

With ``use_degrees=True`` the arm joints come in degrees (``CommandSelector``'s
default ``norm="degrees"``). Without it, LeRobot reports them in -100..100: pass
``CommandSelector(..., norm="range_m100_100")`` instead.

Our virtual leaders implement the rest of ``lerobot.teleoperators.Teleoperator``
too (action/feedback features, calibration hooks, context manager), so they
behave like a LeRobot teleoperator wherever one is expected. They don't
subclass it, because LeRobot isn't a Stage 1 dependency.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol, Self

import numpy as np

from so101_vla.conventions import LEROBOT_FEATURE_NAMES, NUM_JOINTS


class CommandSource(Protocol):
    """What CommandSelector needs from a source (satisfied by LeRobot teleoperators)."""

    def get_action(self) -> dict[str, float]: ...


def _as_action(values: np.ndarray) -> dict[str, float]:
    return dict(zip(LEROBOT_FEATURE_NAMES, (float(v) for v in values), strict=True))


class VirtualLeader:
    """Base for simulated leaders, mirroring ``lerobot.teleoperators.Teleoperator``."""

    name = "virtual_so101_leader"

    def __init__(self, id: str | None = None) -> None:
        self.id = id

    # --- Teleoperator interface -----------------------------------------------

    @property
    def action_features(self) -> dict[str, type]:
        return {k: float for k in LEROBOT_FEATURE_NAMES}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}  # a virtual leader has no motors to drive

    @property
    def is_connected(self) -> bool:
        raise NotImplementedError

    @property
    def is_calibrated(self) -> bool:
        return True  # nothing to calibrate: values are generated in calibrated units

    def connect(self, calibrate: bool = True) -> None:
        raise NotImplementedError

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_action(self) -> dict[str, float]:
        raise NotImplementedError

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        raise NotImplementedError

    def __enter__(self) -> Self:
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.disconnect()


class TrajectoryLeader(VirtualLeader):
    """Virtual leader that plays back joint targets (LeRobot units) over time.

    ``clock`` returns the current time in seconds; pass the simulator's clock
    (e.g. ``lambda: robot.data.time``) for deterministic, sim-synchronous
    playback. Playback starts at ``connect()`` and holds the last sample at the end.
    """

    name = "trajectory_leader"

    def __init__(self, times: np.ndarray, values: np.ndarray, clock: Callable[[], float] = time.perf_counter,
                 loop: bool = False, id: str | None = None) -> None:
        super().__init__(id)
        times, values = np.asarray(times, float), np.asarray(values, float)
        if times.ndim != 1 or values.shape != (len(times), NUM_JOINTS) or np.any(np.diff(times) <= 0):
            raise ValueError("expected increasing times (N,) and values (N, 6)")
        self.times = times - times[0]
        self.values = values
        self.clock = clock
        self.loop = loop
        self._t0: float | None = None

    @property
    def duration(self) -> float:
        return float(self.times[-1])

    def connect(self, calibrate: bool = True) -> None:
        self._t0 = self.clock()

    def disconnect(self) -> None:
        self._t0 = None

    @property
    def is_connected(self) -> bool:
        return self._t0 is not None

    def get_action(self) -> dict[str, float]:
        if self._t0 is None:
            raise RuntimeError("TrajectoryLeader is not connected")
        t = self.clock() - self._t0
        if self.loop and self.duration > 0:
            t %= self.duration
        return _as_action([np.interp(t, self.times, self.values[:, j]) for j in range(NUM_JOINTS)])

    # --- constructors ---------------------------------------------------------

    @classmethod
    def hold(cls, values: np.ndarray, clock: Callable[[], float] = time.perf_counter) -> TrajectoryLeader:
        """A source that always commands one pose."""
        v = np.asarray(values, float)
        return cls(np.array([0.0, 1.0]), np.vstack([v, v]), clock)

    @classmethod
    def sweep(cls, center: np.ndarray, amplitude: np.ndarray, period: np.ndarray, duration: float,
              dt: float = 1 / 30, ramp: float = 1.0,
              clock: Callable[[], float] = time.perf_counter) -> TrajectoryLeader:
        """Smooth per-joint sinusoids around ``center`` (all LeRobot units).

        Each joint follows ``center + envelope(t) * amplitude * sin(2 pi t / period)``,
        where the envelope rises smoothly from 0 to 1 over ``ramp`` seconds, so
        the motion starts at rest at ``center`` and then moves every joint at
        once, like an operator moving a leader arm.
        """
        t = np.arange(0.0, duration + dt / 2, dt)
        center, amplitude, period = (np.asarray(x, float) for x in (center, amplitude, period))
        s = np.clip(t / ramp, 0.0, 1.0) if ramp > 0 else np.ones_like(t)
        envelope = 10 * s**3 - 15 * s**4 + 6 * s**5  # min-jerk smoothstep
        values = center + envelope[:, None] * amplitude * np.sin(2 * np.pi * t[:, None] / period)
        return cls(t, values, clock)

    @classmethod
    def from_teleop_log(cls, path: str | Path, clock: Callable[[], float] = time.perf_counter) -> TrajectoryLeader:
        """Replay the leader commands recorded in a ``teleop.npz`` log."""
        data = np.load(path)
        return cls(data["time"], data["leader_lerobot"], clock)


class SliderLeader(VirtualLeader):
    """Virtual leader driven by on-screen sliders (Tk), in LeRobot units.

    The window is pumped from ``get_action()`` (no separate thread or
    mainloop), so call it every control tick from the main thread.
    """

    name = "slider_leader"

    def __init__(self, initial: Mapping[str, float], ranges: Mapping[str, tuple[float, float]],
                 home: Mapping[str, float] | None = None, title: str = "SO-101 virtual leader",
                 id: str | None = None) -> None:
        super().__init__(id)
        self.initial = dict(initial)
        self.ranges = dict(ranges)
        self.home = dict(home or initial)
        self.title = title
        self._root = None
        self._vars: dict[str, object] = {}
        self.closed = False

    @property
    def is_connected(self) -> bool:
        return self._root is not None

    def connect(self, calibrate: bool = True) -> None:
        import tkinter as tk

        root = tk.Tk()
        root.title(self.title)
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        for row, name in enumerate(LEROBOT_FEATURE_NAMES):
            lo, hi = self.ranges[name]
            unit = "0..100" if name == "gripper.pos" else "deg"
            tk.Label(root, text=f"{name} [{unit}]", width=20, anchor="w").grid(row=row, column=0, padx=6)
            var = tk.DoubleVar(value=self.initial[name])
            tk.Scale(root, variable=var, from_=lo, to=hi, resolution=0.5, orient="horizontal",
                     length=420).grid(row=row, column=1, padx=6)
            self._vars[name] = var
        buttons = tk.Frame(root)
        buttons.grid(row=NUM_JOINTS, column=0, columnspan=2, pady=6)
        tk.Button(buttons, text="Home", command=self._go_home).pack(side="left", padx=4)
        tk.Button(buttons, text="Gripper open", command=lambda: self._vars["gripper.pos"].set(100)).pack(side="left", padx=4)
        tk.Button(buttons, text="Gripper close", command=lambda: self._vars["gripper.pos"].set(0)).pack(side="left", padx=4)
        self._root = root
        # Windows tends to open the window behind the MuJoCo viewer: bring it
        # to the front once, then let it behave like a normal window.
        root.lift()
        root.attributes("-topmost", True)
        root.update()
        root.attributes("-topmost", False)
        root.focus_force()

    def _go_home(self) -> None:
        for name, v in self.home.items():
            self._vars[name].set(v)

    def _on_close(self) -> None:
        self.closed = True

    def disconnect(self) -> None:
        if self._root is not None:
            self._root.destroy()
            self._root = None

    def get_action(self) -> dict[str, float]:
        if self._root is None:
            raise RuntimeError("SliderLeader is not connected")
        self._root.update()
        return {name: float(self._vars[name].get()) for name in LEROBOT_FEATURE_NAMES}
