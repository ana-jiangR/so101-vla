"""Headless MuJoCo tests (no viewer, no rendering)."""

import mujoco
import numpy as np
import pytest

from so101_vla.config import load_robot_config
from so101_vla.control import ArmController
from so101_vla.conventions import JOINT_NAMES
from so101_vla.robot.interface import CommandError
from so101_vla.robot.mujoco_robot import MuJoCoRobot


@pytest.fixture
def cfg():
    return load_robot_config("configs/robot/so101_single.yaml")


@pytest.fixture
def ctrl(cfg):
    robot = MuJoCoRobot.from_config(cfg)
    c = ArmController.from_config(robot, cfg)
    c.reset("rest")
    return c


def test_model_loads_with_expected_joints_and_actuators(cfg):
    robot = MuJoCoRobot.from_config(cfg)
    m = robot.model
    assert m.nq == 6 and m.nu == 6
    assert [m.joint(i).name for i in range(m.njnt)] == [f"main/{j}" for j in JOINT_NAMES]
    assert [m.actuator(i).name for i in range(m.nu)] == [f"main/{j}" for j in JOINT_NAMES]
    assert robot.n_substeps == 7 and robot.control_dt == pytest.approx(1 / 30)
    assert robot.model.opt.timestep == pytest.approx(1 / 210)
    assert robot.model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST  # copied from Menagerie arm
    assert robot.model.camera("main/wrist_cam").id >= 0
    lim = robot.joint_limits("main")
    assert lim.lower[5] == pytest.approx(-0.17453, abs=1e-4) and lim.upper[5] == pytest.approx(1.74533, abs=1e-4)


def test_reset_sets_pose_and_zero_velocity(ctrl, cfg):
    s = ctrl.get_state()
    np.testing.assert_allclose(s.arms["main"].positions, cfg.poses["rest"], atol=1e-12)
    np.testing.assert_allclose(s.arms["main"].velocities, 0.0)
    assert s.time == 0.0


def test_rest_pose_holds_still(ctrl, cfg):
    s = ctrl.hold(2.0)
    assert np.abs(s.arms["main"].positions - cfg.poses["rest"]).max() < 0.01
    assert np.abs(s.arms["main"].velocities).max() < 1e-3


@pytest.mark.parametrize("joint", JOINT_NAMES)
@pytest.mark.parametrize("delta", [0.4, -0.4])
def test_each_joint_moves_individually(ctrl, joint, delta):
    # From "home" (LeRobot HOME), +/-0.4 rad on any single joint stays clear of the floor
    # (from "rest", raising shoulder_lift drives the forearm into the floor).
    ctrl.reset("home")
    j = JOINT_NAMES.index(joint)
    start = ctrl.get_state().arms["main"].positions
    goal = float(start[j] + delta)
    ctrl.move_joint("main", joint, goal)
    assert ctrl.wait_until_settled(tol=0.01)
    q = ctrl.get_state().arms["main"].positions
    assert q[j] == pytest.approx(goal, abs=0.01)
    others = np.delete(np.arange(6), j)
    assert np.abs(q[others] - start[others]).max() < 0.01
    assert ctrl.robot.data.ncon == 0


def test_gripper_open_close_changes_jaw_distance(ctrl):
    robot = ctrl.robot
    jaw = robot.model.body("main/moving_jaw_so101_v1").id
    site = robot.model.site("main/gripperframe").id

    def jaw_gap():
        return np.linalg.norm(robot.data.xipos[jaw] - robot.data.site_xpos[site])

    ctrl.open_gripper("main"); ctrl.wait_until_settled()
    open_gap, q_open = jaw_gap(), ctrl.get_state().arms["main"].positions[5]
    ctrl.close_gripper("main"); ctrl.wait_until_settled()
    closed_gap, q_closed = jaw_gap(), ctrl.get_state().arms["main"].positions[5]
    assert q_open > 1.6 and q_closed < -0.1
    assert open_gap > closed_gap + 0.03


def test_trajectory_tracking_error_is_small(ctrl):
    ctrl.move_through("main", [np.array(p) for p in ([0.5, -0.8, 0.8, 0.6, 0.5, 0.8],
                                                      [-0.5, -0.4, 0.4, 0.9, -0.5, 0.2],
                                                      [0.0, -1.6, 1.55, 0.75, 0.0, 0.0])])
    summary = ctrl.log.summary()["main"]
    assert max(summary["max_abs_error_rad"].values()) < 0.05
    assert summary["ticks_position_clipped"] == 0


def test_backend_rejects_bad_targets(cfg):
    robot = MuJoCoRobot.from_config(cfg)
    with pytest.raises(CommandError, match="hard limits"):
        robot.set_targets({"main": np.array([3.0, 0, 0, 0, 0, 0])})
    with pytest.raises(CommandError, match="unknown arm"):
        robot.set_targets({"left": np.zeros(6)})
    with pytest.raises(CommandError, match="NaN"):
        robot.set_targets({"main": np.full(6, np.nan)})
    np.testing.assert_array_equal(robot.data.ctrl, 0.0)  # nothing was written


def test_huge_jump_is_rate_limited_not_executed_instantly(ctrl):
    before = ctrl.get_state().arms["main"].positions
    report = ctrl.command({"main": np.array([1.8, 1.0, -1.0, 1.0, 2.0, 1.7])})
    assert report.filtered["main"].rate_limited.any()
    assert np.abs(report.filtered["main"].target - before).max() <= 1.5 * ctrl.dt + 1e-9


def test_bimanual_config_builds_and_arms_are_independent():
    cfg = load_robot_config("configs/robot/so101_bimanual.yaml")
    robot = MuJoCoRobot.from_config(cfg)
    assert robot.arm_names == ("left", "right") and robot.model.nu == 12
    c = ArmController.from_config(robot, cfg)
    c.reset("rest")
    c.move_joint("left", "shoulder_pan", 0.8)
    c.wait_until_settled()
    s = c.get_state()
    assert s.arms["left"].positions[0] == pytest.approx(0.8, abs=0.01)
    assert s.arms["right"].positions[0] == pytest.approx(0.0, abs=0.01)
    assert robot.ee_position("left")[1] > robot.ee_position("right")[1]


def test_named_poses_have_no_contacts(cfg):
    robot = MuJoCoRobot.from_config(cfg)
    for name in ("rest", "home", "zero"):
        robot.reset({"main": np.asarray(cfg.poses[name])})
        mujoco.mj_forward(robot.model, robot.data)
        assert robot.data.ncon == 0, name


def test_wrist_camera_renders_rgb_and_observation_matches_lerobot_format(ctrl):
    robot = ctrl.robot
    ctrl.reset("home")
    assert robot.camera_names == ("main/wrist_cam",)
    obs = robot.get_observation(cameras=["main/wrist_cam"], width=320, height=240)
    img = obs.pop("main/wrist_cam")
    assert img.shape == (240, 320, 3) and img.dtype == np.uint8 and img.std() > 5  # not a blank frame
    np.testing.assert_allclose([obs[f"{j}.pos"] for j in JOINT_NAMES], [0, -60, 60, 35, 0, 30], atol=1e-3)
    assert set(robot.get_observation()) == {f"{j}.pos" for j in JOINT_NAMES}  # no rendering unless asked
    with pytest.raises(CommandError, match="unknown camera"):
        robot.render_camera("nope")
    with pytest.raises(CommandError, match="offscreen buffer"):
        robot.render_camera("main/wrist_cam", 4000, 3000)
    robot.close()


def test_bimanual_observation_uses_lerobot_left_right_prefixes():
    cfg = load_robot_config("configs/robot/so101_bimanual.yaml")
    with MuJoCoRobot.from_config(cfg) as robot:
        assert robot.camera_names == ("left/wrist_cam", "right/wrist_cam")
        keys = set(robot.get_observation())
        assert keys == {f"{a}_{j}.pos" for a in ("left", "right") for j in JOINT_NAMES}
