"""Stage 3: scripted expert and LeRobotDataset recording (needs the `train` extra)."""

import numpy as np
import pytest

pytest.importorskip("mink")

from so101_vla.config import load_robot_config  # noqa: E402
from so101_vla.control import ArmController  # noqa: E402
from so101_vla.control.ik import ArmIK, top_down_rotation  # noqa: E402
from so101_vla.conventions import LEROBOT_FEATURE_NAMES  # noqa: E402
from so101_vla.policies.scripted import expert_leader  # noqa: E402
from so101_vla.robot.mujoco_robot import MuJoCoRobot  # noqa: E402
from so101_vla.sim.tabletop import PickPlaceTask  # noqa: E402
from so101_vla.teleop import CommandSelector, TrajectoryLeader  # noqa: E402

TASK = "Pick up the red cube and place it in the green box."


@pytest.fixture
def env():
    cfg = load_robot_config("configs/robot/so101_tabletop.yaml")
    robot = MuJoCoRobot.from_config(cfg)
    ctrl = ArmController.from_config(robot, cfg)
    task = PickPlaceTask(robot)
    task.reset(np.asarray(cfg.poses["home"]), seed=0)
    ctrl.sync()
    yield cfg, robot, ctrl, task, ArmIK(robot, "main", cfg.arms[0].model)
    robot.close()


def test_ik_reaches_top_down_grasp_pose(env):
    cfg, robot, ctrl, task, ik = env
    target = np.array([0.24, -0.05, 0.03])
    res = ik.solve(robot.get_state().arms["main"].positions, target, top_down_rotation(0.3))
    assert res.position_error < 1e-3 and res.orientation_error < 1e-2
    lim = robot.joint_limits("main")
    assert np.all(res.q >= lim.lower[:5]) and np.all(res.q <= lim.upper[:5])
    # IK never moves the simulation
    np.testing.assert_allclose(robot.get_state().arms["main"].positions, ctrl.filters["main"].last_target)


@pytest.mark.parametrize("seed", [0, 7])
def test_scripted_expert_completes_the_task_through_the_safety_stack(env, seed):
    cfg, robot, ctrl, task, ik = env
    task.reset(np.asarray(cfg.poses["home"]), seed=seed)
    ctrl.sync()
    leader, plan = expert_leader(task, ik, robot.get_state().arms["main"].positions, np.asarray(cfg.poses["home"]),
                                 cfg.safety, ctrl.dt, clock=lambda: robot.data.time)
    assert max(p["ik_error_mm"] for p in plan) < 15
    leader.connect()
    sel = CommandSelector(ctrl, "main", {"expert": leader})
    sel.select("expert")
    grasped = lifted = False
    for _ in range(int(np.ceil((leader.duration + 0.5) / ctrl.dt))):
        sel.tick()
        s = task.state()
        grasped, lifted = grasped or s.grasped, lifted or s.lifted
    assert grasped and lifted and task.state().in_box  # detected by the LeRobot env definition
    assert ctrl.log.summary()["main"]["ticks_position_clipped"] == 0  # expert stays within limits


def test_recorder_writes_a_loadable_lerobot_dataset(env, tmp_path):
    pytest.importorskip("lerobot.datasets.lerobot_dataset")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from so101_vla.recording import DemoRecorder, RecordingConfig

    cfg, robot, ctrl, task, ik = env
    rec = DemoRecorder(RecordingConfig(root=tmp_path / "ds", image_size=(32, 24)))
    home_lr = np.array([0.0, -60.0, 60.0, 35.0, 0.0, 30.0])
    first_state = None
    for ep in range(2):
        task.reset(np.asarray(cfg.poses["home"]), seed=ep)  # includes LeRobot's +-2 deg joint noise
        ctrl.sync()
        if first_state is None:
            obs = robot.get_observation()
            first_state = np.array([obs[k] for k in LEROBOT_FEATURE_NAMES])
        leader = TrajectoryLeader(np.array([0.0, 1.0]), np.array([home_lr, home_lr + 10]),
                                  clock=lambda: robot.data.time)
        leader.connect()
        sel = CommandSelector(ctrl, "main", {"src": leader})
        sel.select("src")
        for _ in range(12):
            rec.step(robot, sel, TASK)
        rec.save_episode()
    rec.finalize()
    ds = LeRobotDataset("local/so101_pick_place_sim", root=tmp_path / "ds")
    assert ds.num_episodes == 2 and ds.num_frames == 24 and ds.fps == 30
    assert ds.features["observation.state"]["names"] == list(LEROBOT_FEATURE_NAMES)
    assert ds.features["action"]["names"] == list(LEROBOT_FEATURE_NAMES)
    assert {"observation.images.top", "observation.images.wrist", "observation.images.front"} <= set(ds.features)
    frame = ds[5]
    assert tuple(frame["observation.images.top"].shape) == (3, 24, 32) and frame["task"] == TASK
    # state at tick k is the measured position before the action of tick k (lerobot-record order)
    np.testing.assert_allclose(ds[0]["observation.state"].numpy(), first_state, atol=1e-3)
