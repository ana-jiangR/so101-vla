"""Stage 2 scene (Tasks 2.1, 2.2): cameras, seeded reset, task measurements."""

import mujoco
import numpy as np
import pytest

from so101_vla.config import load_robot_config
from so101_vla.control import ArmController
from so101_vla.robot.mujoco_robot import MuJoCoRobot
from so101_vla.sim.scene import look_at_xyaxes
from so101_vla.sim.tabletop import PickPlaceTask


@pytest.fixture
def env():
    cfg = load_robot_config("configs/robot/so101_tabletop.yaml")
    robot = MuJoCoRobot.from_config(cfg)
    ctrl = ArmController.from_config(robot, cfg)
    task = PickPlaceTask(robot)
    yield cfg, robot, ctrl, task
    robot.close()


def test_scene_has_official_cameras_plus_configured_head(env):
    cfg, robot, *_ = env
    assert robot.camera_names == ("front", "top", "head", "main/wrist_cam")
    assert robot.n_substeps == 7  # same physics as LeRobot's env
    for cam in robot.camera_names:
        img = robot.render_camera(cam, 320, 240)
        assert img.shape == (240, 320, 3) and img.std() > 5


def test_look_at_matches_mujoco_camera_convention():
    # MuJoCo docs: cameras look along -Z of the camera frame; +X right, +Y up.
    pos, target = np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 1.0])
    xy = np.array(look_at_xyaxes(pos, target)).reshape(2, 3)
    z = np.cross(xy[0], xy[1])
    np.testing.assert_allclose(-z, [1, 0, 0], atol=1e-12)  # looks at the target
    np.testing.assert_allclose(xy[1], [0, 0, 1], atol=1e-12)  # image up = world up


def test_head_camera_looks_at_its_target(env):
    cfg, robot, *_ = env
    cam = cfg.cameras[0]
    mujoco.mj_forward(robot.model, robot.data)
    view = -robot.data.cam_xmat[robot.model.camera("head").id].reshape(3, 3)[:, 2]
    expected = np.subtract(cam.target, cam.pos)
    np.testing.assert_allclose(view, expected / np.linalg.norm(expected), atol=1e-9)


def test_seeded_reset_is_reproducible_and_varies(env):
    cfg, robot, ctrl, task = env
    home = np.array(cfg.poses["home"])
    a = task.reset(home, seed=3).cube_pos.copy(); la = dict(task.layout)
    b = task.reset(home, seed=3).cube_pos.copy(); lb = dict(task.layout)
    c = task.reset(home, seed=4).cube_pos.copy()
    np.testing.assert_array_equal(a, b); assert la == lb
    assert np.linalg.norm(a - c) > 1e-3
    for seed in range(20):  # cube always starts on the table, outside the box, untouched
        s = task.reset(home, seed=seed)
        assert not s.in_box and not s.touching and abs(s.cube_height) < 1e-3


def test_cube_in_box_detection_matches_lerobot_definition(env):
    cfg, robot, ctrl, task = env
    task.reset(np.array(cfg.poses["home"]), seed=0)
    ctrl.sync()
    box = robot.data.xpos[robot.model.body("box").id]
    q = task._cube_qadr
    robot.data.qpos[q:q + 3] = [box[0], box[1], 0.03]  # drop the cube into the box
    robot.data.qvel[:] = 0
    ctrl.hold(1.0)
    s = task.state()
    assert s.in_box and s.cube_height < 0.01  # resting on the box floor
    robot.data.qpos[q:q + 3] = [box[0], box[1], 0.12]  # held above the box: not "in" (z >= 6 cm)
    mujoco.mj_forward(robot.model, robot.data)
    assert not task.cube_in_box()


def test_arm_and_objects_do_not_move_while_holding(env):
    cfg, robot, ctrl, task = env
    s0 = task.reset(np.array(cfg.poses["home"]), seed=1); ctrl.sync()
    ctrl.hold(2.0)
    s1 = task.state()
    np.testing.assert_allclose(s1.cube_pos, s0.cube_pos, atol=1e-3)
    np.testing.assert_allclose(robot.get_state().arms["main"].positions, ctrl.filters["main"].last_target, atol=0.01)
