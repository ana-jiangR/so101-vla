"""Stage 2 tabletop task: pick up the cube and place it in the box (Task 2.2).

Built on LeRobot's official SO-101 tabletop scene (assets/lerobot_tabletop/). The
reset randomization and the "cube in box" test follow LeRobot's env
(huggingface.co/lerobot/so101-tabletop-mujoco, ``env.py``: ``DEFAULTS["randomize"]``,
``_World.reset`` / ``cubes_in_box``), restricted to one cube. Grasp/lift checks and
the distance metric are ours.

The robot itself is the Stage 1 ``MuJoCoRobot``: motion still only goes through its
actuators via ``ArmController``. This module only places objects and measures.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from so101_vla.robot.mujoco_robot import MuJoCoRobot
from so101_vla.sim.scene import arm_prefix

# LeRobot env: a cube is in the box when its center is inside the box's inner
# footprint (box frame) and below 6 cm, i.e. resting in it, not held above it.
BOX_INNER_HALF = 0.045
IN_BOX_MAX_HEIGHT = 0.06
CUBE_HALF = 0.015
# Gripper bodies of the Menagerie arm whose geoms count as "touching" the cube.
GRIPPER_BODIES = ("gripper", "moving_jaw_so101_v1")
MOVING_JAW_BODY = "moving_jaw_so101_v1"


@dataclass(frozen=True)
class Randomization:
    """Seeded reset offsets (half-ranges). Defaults are LeRobot's tabletop env values."""

    cube_pos: tuple[float, float] = (0.06, 0.14)  # m, x/y around the XML pose
    cube_yaw: float = np.pi  # rad
    box_pos: tuple[float, float] = (0.03, 0.015)  # m
    joint_noise_deg: float = 2.0  # deg (gripper: 0..100 units), as LeRobot's "joints"
    camera_pos: dict[str, float] = field(default_factory=lambda: {"front": 0.01})  # m per axis
    min_cube_box_gap: float = 0.01  # m between footprints (LeRobot: no overlap)


@dataclass(frozen=True)
class TaskState:
    """What the evaluation needs at one instant (world frame, metres)."""

    time: float
    cube_pos: np.ndarray
    gripper_pos: np.ndarray
    gripper_to_cube: float
    cube_height: float  # of the cube's bottom face above the table
    touching: bool  # any gripper geom touches the cube
    grasped: bool  # both jaws touch the cube
    lifted: bool  # grasped, off the table and >= 2 cm up
    in_box: bool  # LeRobot env definition
    cube_on_table_contact: bool


class PickPlaceTask:
    """One cube, one box, one arm on the tabletop."""

    def __init__(self, robot: MuJoCoRobot, arm: str = "main", cube: str = "red_cube", box: str = "box",
                 randomization: Randomization | None = None) -> None:
        self.robot, self.arm = robot, arm
        self.randomization = randomization or Randomization()
        m = robot.model
        self._cube = m.body(cube).id
        self._box = m.body(box).id
        jnt = m.body_jntadr[self._cube]
        if m.body_jntnum[self._cube] != 1 or m.jnt_type[jnt] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"cube body {cube!r} must have exactly one freejoint")
        self._cube_qadr = int(m.jnt_qposadr[jnt])
        self._cube_dof = int(m.jnt_dofadr[jnt])
        self._cube_geoms = set(np.flatnonzero(m.geom_bodyid == self._cube).tolist())
        prefix = arm_prefix(arm)
        grip_bodies = {m.body(prefix + b).id for b in GRIPPER_BODIES}
        self._grip_geoms = set(np.flatnonzero(np.isin(m.geom_bodyid, list(grip_bodies))).tolist())
        self._moving_jaw_geoms = set(np.flatnonzero(m.geom_bodyid == m.body(prefix + MOVING_JAW_BODY).id).tolist())
        self._fixed_jaw_geoms = self._grip_geoms - self._moving_jaw_geoms
        self._table = m.geom("table").id
        self._ee_site = m.site(prefix + "gripperframe").id
        # XML poses to randomize around.
        self._cube_qpos0 = m.qpos0[self._cube_qadr:self._cube_qadr + 7].copy()
        self._box_pos0 = m.body_pos[self._box].copy()
        self._cam_pos0 = m.cam_pos.copy()
        tid =mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TEXT, "instruction")
        self.scene_instruction = (m.text_data[m.text_adr[tid]:m.text_adr[tid] + m.text_size[tid] - 1].decode()
                                  if tid >= 0 else "")
        self.layout: dict = {}

    # --- reset ------------------------------------------------------------------

    def reset(self, arm_pose: np.ndarray, seed: int | None = None) -> TaskState:
        """Put the arm at ``arm_pose`` [rad] (+ joint noise) and place cube/box with seeded offsets.

        The arm pose goes through ``MuJoCoRobot.reset`` (validated). Objects are set
        directly: they are not actuated, and this only happens at reset.
        """
        rng = np.random.default_rng(seed)
        r, m, d = self.robot, self.robot.model, self.robot.data
        rz = self.randomization

        noise = np.deg2rad(rng.uniform(-rz.joint_noise_deg, rz.joint_noise_deg, 6))
        lim = r.joint_limits(self.arm)
        noise[5] = rng.uniform(-rz.joint_noise_deg, rz.joint_noise_deg) / 100.0 * (lim.upper[5] - lim.lower[5])
        pose = np.clip(np.asarray(arm_pose, float) + noise, lim.lower, lim.upper)

        m.cam_pos[:] = self._cam_pos0
        m.body_pos[self._box] = self._box_pos0
        r.reset({self.arm: pose})  # mj_resetData + validated arm pose

        box_xy = self._box_pos0[:2] + rng.uniform(-1, 1, 2) * rz.box_pos
        m.body_pos[self._box, :2] = box_xy
        box_half = BOX_INNER_HALF + 0.004 + 0.002  # outer half-size incl. walls
        for _ in range(100):
            cube_xy = self._cube_qpos0[:2] + rng.uniform(-1, 1, 2) * rz.cube_pos
            if np.max(np.abs(cube_xy - box_xy)) > box_half + CUBE_HALF * np.sqrt(2) + rz.min_cube_box_gap:
                break
        else:
            raise RuntimeError("could not place the cube outside the box")
        yaw = rng.uniform(-rz.cube_yaw, rz.cube_yaw)
        q = self._cube_qadr
        d.qpos[q:q + 2] = cube_xy
        d.qpos[q + 2] = self._cube_qpos0[2]
        d.qpos[q + 3:q + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        d.qvel[self._cube_dof:self._cube_dof + 6] = 0.0
        for name, half in rz.camera_pos.items():
            cid = m.camera(name).id
            m.cam_pos[cid] = self._cam_pos0[cid] + rng.uniform(-half, half, 3)
        mujoco.mj_forward(m, d)
        self.layout = {"seed": seed, "cube_xy": cube_xy.round(4).tolist(), "cube_yaw": round(float(yaw), 4),
                       "box_xy": box_xy.round(4).tolist(), "arm_pose": pose.round(4).tolist()}
        return self.state()

    # --- measurements -------------------------------------------------------------

    def _contacts(self) -> tuple[set[int], bool]:
        d = self.robot.data
        touching: set[int] = set()
        on_table = False
        for i in range(d.ncon):
            g1, g2 = int(d.contact[i].geom1), int(d.contact[i].geom2)
            for a, b in ((g1, g2), (g2, g1)):
                if a in self._cube_geoms:
                    if b in self._grip_geoms:
                        touching.add(b)
                    elif b == self._table:
                        on_table = True
        return touching, on_table

    def cube_in_box(self) -> bool:
        d = self.robot.data
        center, rot = d.xpos[self._box], d.xmat[self._box].reshape(3, 3)
        local = (d.xpos[self._cube] - center) @ rot
        return bool(np.all(np.abs(local[:2]) < BOX_INNER_HALF) and local[2] < IN_BOX_MAX_HEIGHT)

    def state(self) -> TaskState:
        d = self.robot.data
        cube = d.xpos[self._cube].copy()
        grip = d.site_xpos[self._ee_site].copy()
        touching, on_table = self._contacts()
        grasped = bool(touching & self._moving_jaw_geoms) and bool(touching & self._fixed_jaw_geoms)
        height = float(cube[2] - CUBE_HALF)
        return TaskState(
            time=float(d.time), cube_pos=cube, gripper_pos=grip,
            gripper_to_cube=float(np.linalg.norm(grip - cube)), cube_height=height,
            touching=bool(touching), grasped=grasped, lifted=grasped and not on_table and height > 0.02,
            in_box=self.cube_in_box(), cube_on_table_contact=on_table,
        )
