"""Compose a MuJoCo model from a scene file plus one or more SO-101 arms.

Follows the approach of LeRobot's official SO-101 MuJoCo environment
(huggingface.co/lerobot/so101-tabletop-mujoco, ``env.py: build_model``):
load the scene and the Menagerie arm as separate MjSpecs, attach the arm, and,
if the scene has no ``<option>`` of its own, use the arm's physics options.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path

import mujoco
import numpy as np

from so101_vla.config import ArmConfig, CameraConfig, RobotConfig

# Physics options copied from the arm MJCF (Menagerie: implicitfast, elliptic
# cones, impratio 10, ...) when the scene does not define <option>.
_OPTION_FIELDS = ("timestep", "integrator", "cone", "jacobian", "solver", "iterations",
                  "ls_iterations", "noslip_iterations", "impratio", "tolerance", "ls_tolerance")


def arm_prefix(arm_name: str) -> str:
    """Name prefix for every element of an attached arm, e.g. 'main/elbow_flex'."""
    return f"{arm_name}/"


def _has_option(xml_path: Path) -> bool:
    return ET.parse(xml_path).getroot().find("option") is not None


def build_spec(scene_xml: Path, arms: Sequence[ArmConfig]) -> mujoco.MjSpec:
    """Attach each arm's (unmodified) MJCF to the scene under its prefix.

    Attaching via MjSpec, rather than <include>, keeps each arm's mesh paths
    relative to its own file and lets several copies coexist without name
    clashes.
    """
    spec = mujoco.MjSpec.from_file(str(scene_xml))
    copy_options = not _has_option(scene_xml)
    for i, arm in enumerate(arms):
        arm_spec = mujoco.MjSpec.from_file(str(arm.model))
        if arm.color is not None:  # as LeRobot's tabletop env: recolor printed parts, keep motors
            for material in arm_spec.materials:
                if not material.name.startswith("sts3215"):
                    material.rgba = [*arm.color, 1.0]
        if copy_options and i == 0:
            for name in _OPTION_FIELDS:
                setattr(spec.option, name, getattr(arm_spec.option, name))
        frame = spec.worldbody.add_frame(pos=list(arm.mount_pos), quat=list(arm.mount_quat))
        spec.attach(arm_spec, prefix=arm_prefix(arm.name), frame=frame)
    return spec


def look_at_xyaxes(pos: Sequence[float], target: Sequence[float]) -> list[float]:
    """MuJoCo camera ``xyaxes`` for a camera at ``pos`` looking at ``target`` with world z up.

    A MuJoCo camera looks along its -z axis; x points right and y up in the image.
    """
    forward = np.asarray(target, float) - np.asarray(pos, float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    if np.linalg.norm(right) < 1e-6:
        raise ValueError("camera looks straight up or down; give the target an x/y offset")
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return [*right.tolist(), *up.tolist()]


def add_cameras(spec: mujoco.MjSpec, cameras: Sequence[CameraConfig]) -> None:
    """Add fixed world cameras and grow the offscreen buffer to fit their render size."""
    existing = {c.name for c in spec.cameras}
    for cam in cameras:
        if cam.name in existing:
            raise ValueError(f"camera {cam.name!r} already exists in the scene")
        spec.worldbody.add_camera(name=cam.name, pos=list(cam.pos), fovy=cam.fovy,
                                  xyaxes=look_at_xyaxes(cam.pos, cam.target))
        g = spec.visual.global_
        g.offwidth, g.offheight = max(g.offwidth, cam.width), max(g.offheight, cam.height)


def build_model(config: RobotConfig) -> mujoco.MjModel:
    spec = build_spec(config.scene, config.arms)
    add_cameras(spec, config.cameras)
    model = spec.compile()
    if config.physics_timestep is not None:
        model.opt.timestep = config.physics_timestep
    return model
