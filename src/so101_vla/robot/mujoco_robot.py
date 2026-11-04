"""MuJoCo backend for RobotInterface."""

from __future__ import annotations

from collections.abc import Mapping

import mujoco
import numpy as np

from so101_vla.config import RobotConfig
from so101_vla.conventions import EE_SITE, JOINT_NAMES
from so101_vla.robot.interface import (
    ArmState,
    CommandError,
    JointLimits,
    RobotInterface,
    RobotState,
    check_target_array,
)
from so101_vla.sim.scene import arm_prefix, build_model


class _ArmIndex:
    """Index arrays into qpos/qvel/ctrl for one arm, in canonical joint order."""

    def __init__(self, model: mujoco.MjModel, arm: str) -> None:
        prefix = arm_prefix(arm)
        joint_ids = [model.joint(prefix + j).id for j in JOINT_NAMES]
        self.qpos = np.array([model.jnt_qposadr[j] for j in joint_ids])
        self.qvel = np.array([model.jnt_dofadr[j] for j in joint_ids])
        self.ctrl = np.array([model.actuator(prefix + j).id for j in JOINT_NAMES])
        self.ee_site = model.site(prefix + EE_SITE).id

        for joint, jid, aid in zip(JOINT_NAMES, joint_ids, self.ctrl, strict=True):
            _check_position_actuator(model, aid, jid, f"{prefix}{joint}")

        # Usable range = joint range intersected with actuator ctrlrange.
        jr = model.jnt_range[joint_ids]
        cr = model.actuator_ctrlrange[self.ctrl]
        self.limits = JointLimits(
            lower=np.maximum(jr[:, 0], cr[:, 0]),
            upper=np.minimum(jr[:, 1], cr[:, 1]),
        )


def _check_position_actuator(model: mujoco.MjModel, aid: int, jid: int, name: str) -> None:
    """Verify ``ctrl`` really is a joint-position target for this actuator.

    A MuJoCo <position> actuator computes force = kp*ctrl - kp*q - kv*qdot,
    i.e. gainprm[0] = kp and biasprm[:3] = [0, -kp, -kv], acting on its joint
    with gear 1. If the model is changed so this no longer holds, writing joint
    angles into ctrl would be wrong, so fail loudly.
    """
    kp = model.actuator_gainprm[aid, 0]
    bias = model.actuator_biasprm[aid, :3]
    ok = (
        model.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_JOINT
        and model.actuator_trnid[aid, 0] == jid
        and model.actuator_gaintype[aid] == mujoco.mjtGain.mjGAIN_FIXED
        and model.actuator_biastype[aid] == mujoco.mjtBias.mjBIAS_AFFINE
        and np.isclose(model.actuator_gear[aid, 0], 1.0)
        and kp > 0
        and np.isclose(bias[0], 0.0)
        and np.isclose(bias[1], -kp)
        and model.actuator_ctrllimited[aid]
    )
    if not ok:
        raise ValueError(f"actuator {name!r} is not a gear-1, ctrl-limited position actuator on its joint")


class MuJoCoRobot(RobotInterface):
    """SO-101 arm(s) simulated in MuJoCo, driven through their position actuators.

    ``step()`` holds the current targets for ``n_substeps`` physics steps, so
    the control rate (e.g. 30 Hz) is decoupled from the physics rate.

    As in LeRobot's SO-101 MuJoCo env, the physics timestep is adjusted
    slightly so one control period is a whole number of physics steps:
    ``n = round(1 / (hz * timestep))``, ``timestep = 1 / (hz * n)``
    (Menagerie's 0.005 s at 30 Hz -> 7 substeps of 1/210 s).
    """

    def __init__(self, model: mujoco.MjModel, arm_names: tuple[str, ...], control_hz: float) -> None:
        if control_hz <= 0:
            raise ValueError("control_hz must be positive")
        self.n_substeps = max(1, round(1.0 / (control_hz * model.opt.timestep)))
        model.opt.timestep = 1.0 / (control_hz * self.n_substeps)
        self._control_dt = 1.0 / control_hz

        self.model = model
        self.data = mujoco.MjData(model)
        self._arm_names = tuple(arm_names)
        self._idx = {arm: _ArmIndex(model, arm) for arm in self._arm_names}
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}

    @classmethod
    def from_config(cls, config: RobotConfig) -> MuJoCoRobot:
        return cls(build_model(config), config.arm_names, config.control_hz)

    # --- RobotInterface -------------------------------------------------------

    @property
    def arm_names(self) -> tuple[str, ...]:
        return self._arm_names

    @property
    def control_dt(self) -> float:
        return self._control_dt

    def joint_limits(self, arm: str) -> JointLimits:
        return self._index(arm).limits

    def get_state(self) -> RobotState:
        d = self.data
        arms = {
            arm: ArmState(positions=d.qpos[ix.qpos].copy(), velocities=d.qvel[ix.qvel].copy())
            for arm, ix in self._idx.items()
        }
        return RobotState(time=float(d.time), arms=arms)

    def get_targets(self) -> dict[str, np.ndarray]:
        return {arm: self.data.ctrl[ix.ctrl].copy() for arm, ix in self._idx.items()}

    def set_targets(self, targets: Mapping[str, np.ndarray]) -> None:
        checked = {}
        for arm, target in targets.items():
            ix = self._index(arm)
            q = check_target_array(target, arm)
            inside = ix.limits.contains(q)
            if not inside.all():
                bad = [f"{JOINT_NAMES[i]}={q[i]:+.4f} not in [{ix.limits.lower[i]:+.4f}, {ix.limits.upper[i]:+.4f}]"
                       for i in np.flatnonzero(~inside)]
                raise CommandError(f"arm {arm!r}: target outside hard limits: {', '.join(bad)}")
            checked[arm] = q
        # Only write once every arm's target is valid.
        for arm, q in checked.items():
            self.data.ctrl[self._idx[arm].ctrl] = q

    def step(self) -> RobotState:
        mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
        return self.get_state()

    def reset(self, pose: Mapping[str, np.ndarray]) -> RobotState:
        """Teleport to ``pose`` with zero velocity (simulation only)."""
        missing = set(self._arm_names) - set(pose)
        if missing:
            raise CommandError(f"reset pose missing arms: {sorted(missing)}")
        mujoco.mj_resetData(self.model, self.data)
        self.set_targets(pose)  # validates
        for arm, q in pose.items():
            self.data.qpos[self._idx[arm].qpos] = q
        mujoco.mj_forward(self.model, self.data)
        return self.get_state()

    @property
    def camera_names(self) -> tuple[str, ...]:
        """MuJoCo camera names, e.g. ``"main/wrist_cam"`` (arm cameras carry the arm prefix)."""
        return tuple(self.model.camera(i).name for i in range(self.model.ncam))

    def render_camera(self, name: str, width: int = 640, height: int = 480) -> np.ndarray:
        """RGB image (height, width, 3) uint8 from a named MuJoCo camera, using ``mujoco.Renderer``.

        Size is capped by the scene's ``<visual><global offwidth/offheight>``
        (1280x960 in ``arm_only.xml``). One renderer is kept per resolution.
        """
        if name not in self.camera_names:
            raise CommandError(f"unknown camera {name!r}; robot has {self.camera_names}")
        vis = self.model.vis.global_
        if width > vis.offwidth or height > vis.offheight:
            raise CommandError(f"camera size {width}x{height} exceeds offscreen buffer "
                               f"{vis.offwidth}x{vis.offheight}; raise offwidth/offheight in the scene XML")
        renderer = self._renderers.get((width, height))
        if renderer is None:
            renderer = self._renderers[(width, height)] = mujoco.Renderer(self.model, height, width)
        renderer.update_scene(self.data, camera=name)
        return renderer.render().copy()

    def close(self) -> None:
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()

    # --- MuJoCo-specific helpers ----------------------------------------------

    def ee_position(self, arm: str) -> np.ndarray:
        """World position of the arm's gripper frame site [m]."""
        return self.data.site_xpos[self._index(arm).ee_site].copy()

    def _index(self, arm: str) -> _ArmIndex:
        try:
            return self._idx[arm]
        except KeyError:
            raise CommandError(f"unknown arm {arm!r}; robot has {self._arm_names}") from None
