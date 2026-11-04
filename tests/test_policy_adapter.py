"""Stage 2 action path without a GPU: chunk validation and the policy as a command source."""

import numpy as np
import pytest

from so101_vla.config import load_robot_config
from so101_vla.control import ArmController
from so101_vla.conventions import LeRobotUnits
from so101_vla.policies import ActionChunkValidator, InvalidChunk, PolicySource
from so101_vla.robot.mujoco_robot import MuJoCoRobot
from so101_vla.teleop import CommandSelector, TrajectoryLeader

HOME_LR = np.array([0.0, -60.0, 60.0, 35.0, 0.0, 30.0])


@pytest.fixture
def setup():
    cfg = load_robot_config("configs/robot/so101_tabletop.yaml")
    robot = MuJoCoRobot.from_config(cfg)
    ctrl = ArmController.from_config(robot, cfg)
    ctrl.reset("home")
    lim = robot.joint_limits("main")
    yield robot, ctrl, ActionChunkValidator(LeRobotUnits(lim.lower, lim.upper))
    robot.close()


class FakePolicy:
    """Stands in for SmolVLAPolicy: returns a fixed chunk, records what it saw."""

    camera_names = ["top"]
    last_inference_ms = 1.0

    def __init__(self, chunk):
        self.chunk, self.calls, self.instructions = np.asarray(chunk, float), 0, []

    def reset(self):
        pass

    def predict_chunk(self, observation, instruction):
        assert observation["top"].dtype == np.uint8 and "shoulder_pan.pos" in observation
        self.calls += 1
        self.instructions.append(instruction)
        return self.chunk


def _source(robot, ctrl, validator, policy, **kw):
    return PolicySource(policy, observe=lambda: robot.get_observation(cameras=policy.camera_names, width=64, height=48),
                        clock=lambda: robot.data.time, current_q=lambda: robot.get_state().arms["main"].positions,
                        validator=validator, instruction="Pick up the cube.", **kw)


def test_validator_reports_range_and_jumps(setup):
    robot, ctrl, v = setup
    chunk = np.tile(HOME_LR, (5, 1)); chunk[2, 1] = -150.0  # shoulder_lift beyond -100 deg
    r = v.validate(chunk, robot.get_state().arms["main"].positions)
    assert r.steps == 5 and r.out_of_range_steps["shoulder_lift"] == 1 and r.any_out_of_range
    assert r.max_excess["shoulder_lift"] == pytest.approx(50.0, abs=0.1)
    assert r.max_step_change["shoulder_lift"] == pytest.approx(90.0)
    assert r.first_step_distance["shoulder_pan"] < 1e-6


@pytest.mark.parametrize("bad", [np.zeros((5, 5)), np.zeros(6), np.zeros((0, 6)),
                                 np.array([[0, 0, np.nan, 0, 0, 0]])])
def test_validator_rejects_malformed_chunks(setup, bad):
    robot, ctrl, v = setup
    with pytest.raises(InvalidChunk):
        v.validate(bad, np.zeros(6))


def test_policy_chunks_are_executed_through_selector_and_replanned(setup):
    robot, ctrl, v = setup
    target = HOME_LR + np.array([10, 5, -5, 5, 10, 20])
    policy = FakePolicy(np.tile(target, (10, 1)))
    src = _source(robot, ctrl, v, policy)
    sel = CommandSelector(ctrl, "main", {"policy": src})
    sel.select("policy"); sel.run(45 * ctrl.dt)
    assert policy.calls == 5  # 45 ticks / 10-step chunks: re-observed when each chunk runs out
    assert len(src.executed) == 45 and src.arrays()["frames/top"].shape == (5, 48, 64, 3)
    lim = robot.joint_limits("main")
    np.testing.assert_allclose(robot.get_state().arms["main"].positions,
                               LeRobotUnits(lim.lower, lim.upper).from_lerobot(target), atol=0.02)
    src.reset(); src.replan_every = 4
    sel.select("policy"); sel.run(8 * ctrl.dt)
    assert policy.calls == 7  # replan every 4 actions


def test_out_of_range_policy_actions_are_clipped_by_safety_filter(setup):
    robot, ctrl, v = setup
    bad = np.tile(HOME_LR, (30, 1)); bad[:, 1] = -400.0  # far past shoulder_lift's -100 deg limit
    src = _source(robot, ctrl, v, FakePolicy(bad))
    sel = CommandSelector(ctrl, "main", {"policy": src})
    sel.select("policy"); sel.run(30 * ctrl.dt)
    q = robot.get_state().arms["main"].positions
    assert q[1] >= robot.joint_limits("main").lower[1] - 0.01
    assert ctrl.log.summary()["main"]["ticks_position_clipped"] > 0
    assert src.chunks[0].report.any_out_of_range


def test_invalid_chunk_makes_selector_hold(setup):
    robot, ctrl, v = setup
    src = _source(robot, ctrl, v, FakePolicy(np.full((10, 6), np.nan)))
    sel = CommandSelector(ctrl, "main", {"policy": src})
    before = robot.get_state().arms["main"].positions
    sel.select("policy"); sel.run(10 * ctrl.dt)
    assert sel.active is None and any("InvalidChunk" in e.get("error", "") for e in sel.log.events)
    np.testing.assert_allclose(robot.get_state().arms["main"].positions, before, atol=0.01)


def test_switching_from_teleop_to_policy_is_safe(setup):
    robot, ctrl, v = setup
    leader = TrajectoryLeader.hold(HOME_LR, lambda: robot.data.time); leader.connect()
    far = np.tile(HOME_LR + np.array([40, 20, -20, 0, 0, 50]), (50, 1))
    src = _source(robot, ctrl, v, FakePolicy(far))
    sel = CommandSelector(ctrl, "main", {"teleop": leader, "policy": src})
    sel.select("teleop"); sel.run(0.5)
    sel.select("policy"); sel.run(1.0)
    steps = np.abs(np.diff(np.asarray(sel.log.commanded), axis=0)).max()
    assert steps <= sel.engage.engage_velocity * ctrl.dt + 1e-9


def test_run_episode_reports_three_levels_and_saves_logs(setup, tmp_path):
    from so101_vla.evaluation import EpisodeConfig, run_episode
    from so101_vla.sim.tabletop import PickPlaceTask

    robot, ctrl, v = setup
    task = PickPlaceTask(robot)
    src = _source(robot, ctrl, v, FakePolicy(np.tile(HOME_LR + [15, 0, 0, 0, 0, 0], (20, 1))))
    res = run_episode(robot, ctrl, task, src, EpisodeConfig("Pick up the cube.", seed=0, max_seconds=2.0), tmp_path)
    assert res["level1_inference"] and res["level2_control"] and not res["level3_success"]
    assert res["stop"] == "timeout" and res["chunks"] == 3 and res["actions_executed"] == 60
    for f in ("episode.json", "policy.npz", "task_state.npz", "teleop.npz", "tracking.npz", "first_top.png"):
        assert (tmp_path / f).is_file()
    bad = _source(robot, ctrl, v, FakePolicy(np.full((5, 6), np.inf)))
    res = run_episode(robot, ctrl, task, bad, EpisodeConfig("x", seed=1, max_seconds=1.0))
    assert res["stop"] == "source_error" and not res["level1_inference"] and not res["level2_control"]
