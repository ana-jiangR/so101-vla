"""Task 1.6: virtual leader -> simulated follower (headless)."""

import numpy as np
import pytest

from so101_vla.config import load_robot_config
from so101_vla.control import ArmController
from so101_vla.conventions import LEROBOT_FEATURE_NAMES, from_lerobot, to_lerobot
from so101_vla.robot.mujoco_robot import MuJoCoRobot
from so101_vla.teleop import CommandSelector, EngageConfig, TrajectoryLeader, estimate_lag_ms

G_LO, G_HI = -0.174533, 1.7453292
HOME_LR = np.array([0.0, -60.0, 60.0, 35.0, 0.0, 30.0])  # LeRobot env HOME


@pytest.fixture
def setup():
    cfg = load_robot_config("configs/robot/so101_single.yaml")
    robot = MuJoCoRobot.from_config(cfg)
    ctrl = ArmController.from_config(robot, cfg)
    ctrl.reset("home")
    clock = lambda: robot.data.time  # noqa: E731
    return robot, ctrl, clock


def test_every_joint_and_gripper_mapped(setup):
    robot, ctrl, clock = setup
    target_lr = HOME_LR + np.array([20.0, 8.0, -10.0, 12.0, 30.0, 40.0])  # distinct offset per joint
    leader = TrajectoryLeader.hold(target_lr, clock)
    leader.connect()
    sel = CommandSelector(ctrl, "main", {"leader": leader})
    sel.select("leader")
    sel.run(3.0)
    q = robot.get_state().arms["main"].positions
    np.testing.assert_allclose(q, from_lerobot(target_lr, G_LO, G_HI), atol=0.01)
    np.testing.assert_allclose(to_lerobot(q, G_LO, G_HI), target_lr, atol=0.6)  # deg / gripper units


def test_safe_engage_from_far_pose_is_slow_and_jump_free(setup):
    robot, ctrl, clock = setup
    ctrl.reset("rest")
    leader = TrajectoryLeader.hold(HOME_LR, clock)
    leader.connect()
    engage = EngageConfig(engage_velocity=0.8, engage_tolerance=0.03)
    sel = CommandSelector(ctrl, "main", {"leader": leader}, engage)
    sel.select("leader")
    sel.run(4.0)
    cmd = np.asarray(sel.log.commanded)
    steps = np.abs(np.diff(cmd, axis=0)).max(axis=1)
    assert steps.max() <= 0.8 * ctrl.dt + 1e-9  # never faster than the engage speed
    assert sel.engaged and any(e.get("engaged") == "leader" for e in sel.log.events)
    assert robot.data.ncon == 0


def test_only_one_source_active_and_switching_is_bounded(setup):
    robot, ctrl, clock = setup
    a = TrajectoryLeader.hold(HOME_LR, clock)
    b_lr = np.array([-25.0, -45.0, 50.0, 60.0, 20.0, 80.0])
    b = TrajectoryLeader.hold(b_lr, clock)
    for s in (a, b):
        s.connect()
    sel = CommandSelector(ctrl, "main", {"teleop": a, "policy": b})
    sel.select("teleop"); sel.run(1.0)
    np.testing.assert_allclose(robot.get_state().arms["main"].positions, from_lerobot(HOME_LR, G_LO, G_HI), atol=0.01)
    sel.select("policy"); sel.run(3.0)
    np.testing.assert_allclose(robot.get_state().arms["main"].positions, from_lerobot(b_lr, G_LO, G_HI), atol=0.01)
    cmd = np.asarray(sel.log.commanded)
    assert np.abs(np.diff(cmd, axis=0)).max() <= sel.engage.engage_velocity * ctrl.dt + 1e-9
    assert set(sel.log.source) == {"teleop", "policy"}
    sel.select(None); sel.run(0.5)  # no source: hold
    np.testing.assert_allclose(robot.get_state().arms["main"].positions, from_lerobot(b_lr, G_LO, G_HI), atol=0.01)
    with pytest.raises(KeyError):
        sel.select("nope")


def test_out_of_range_leader_values_are_clipped_not_executed(setup):
    robot, ctrl, clock = setup
    bad = HOME_LR.copy(); bad[5] = 150.0; bad[0] = 179.0  # gripper > 100, pan past its 110 deg limit
    leader = TrajectoryLeader.hold(bad, clock); leader.connect()
    sel = CommandSelector(ctrl, "main", {"leader": leader})
    sel.select("leader"); sel.run(4.0)
    q = robot.get_state().arms["main"].positions
    hard = robot.joint_limits("main")
    assert q[5] <= hard.upper[5] + 0.01 and q[0] <= hard.upper[0]
    assert ctrl.log.summary()["main"]["ticks_position_clipped"] > 0


def test_moving_leader_tracking_and_lag(setup):
    robot, ctrl, clock = setup
    leader = TrajectoryLeader.sweep(HOME_LR, [30, 10, 10, 15, 45, 25], [6, 7, 5, 4.5, 8, 3], 10.0, clock=clock)
    leader.connect()
    sel = CommandSelector(ctrl, "main", {"leader": leader})
    sel.select("leader"); sel.run(10.0)
    s = sel.log.summary(ctrl.dt)
    assert s["engaged_ticks"] == s["ticks"]  # starts aligned at home
    assert max(s["leader_to_follower_error_rad"]["max_abs"].values()) < 0.05
    lags = [v for v in s["follower_lag_ms"].values() if v is not None]
    assert lags and all(20 <= v <= 70 for v in lags)  # about one 33 ms control period
    assert robot.data.ncon == 0


def test_lag_estimator_on_synthetic_delay():
    dt, n = 1 / 30, 300
    t = np.arange(n) * dt
    leader = np.tile(np.sin(2 * np.pi * t / 3)[:, None], (1, 6))
    measured = np.vstack([leader[:1], leader[:1], leader[:-2]])  # measured[i] = leader[i-2] -> 3 ticks after sampling
    lags = estimate_lag_ms(leader, measured, dt)
    assert all(v == pytest.approx(3 * dt * 1000, abs=2.0) for v in lags.values())


def test_log_replay_roundtrip(setup, tmp_path):
    robot, ctrl, clock = setup
    leader = TrajectoryLeader.sweep(HOME_LR, [20, 5, 5, 10, 30, 20], [4, 5, 6, 3, 7, 2], 3.0, clock=clock)
    leader.connect()
    sel = CommandSelector(ctrl, "main", {"leader": leader})
    sel.select("leader"); sel.run(3.0)
    sel.log.save(tmp_path, ctrl.dt)
    replay = TrajectoryLeader.from_teleop_log(tmp_path / "teleop.npz", clock=lambda: 1.5)
    replay._t0 = 0.0
    original = leader.values[np.searchsorted(leader.times, 1.5)]
    np.testing.assert_allclose([replay.get_action()[k] for k in LEROBOT_FEATURE_NAMES], original, atol=1.0)


def test_failing_source_holds_position_instead_of_crashing(setup):
    robot, ctrl, clock = setup

    class FlakyLeader:  # e.g. a serial read that starts failing
        calls = 0

        def get_action(self):
            self.calls += 1
            if self.calls > 30:
                raise OSError("serial read timeout")
            return dict(zip(LEROBOT_FEATURE_NAMES, HOME_LR + 10.0, strict=True))

    sel = CommandSelector(ctrl, "main", {"flaky": FlakyLeader()})
    sel.select("flaky"); sel.run(1.0)
    before = robot.get_state().arms["main"].positions
    sel.run(1.0)  # source fails from tick 31 on
    assert sel.active is None
    assert any(e.get("source_error") == "flaky" and "serial read timeout" in e["error"] for e in sel.log.events)
    np.testing.assert_allclose(robot.get_state().arms["main"].positions, before, atol=0.02)  # held still
    with pytest.raises(OSError):
        strict = CommandSelector(ctrl, "main", {"flaky": FlakyLeader()}, on_source_error="raise")
        strict.select("flaky"); strict.run(2.0)


def test_lerobot_bimanual_key_prefix(setup):
    """LeRobot bi_so_leader actions carry 'left_'/'right_' prefixes; pick one arm's keys."""
    robot, ctrl, clock = setup
    left, right = HOME_LR + 8.0, HOME_LR - 8.0

    class FakeBiLeader:
        def get_action(self):
            return {**{f"left_{k}": v for k, v in zip(LEROBOT_FEATURE_NAMES, left, strict=True)},
                    **{f"right_{k}": v for k, v in zip(LEROBOT_FEATURE_NAMES, right, strict=True)}}

    sel = CommandSelector(ctrl, "main", {"bi": FakeBiLeader()}, key_prefix="right_")
    sel.select("bi"); sel.run(2.0)
    np.testing.assert_allclose(robot.get_state().arms["main"].positions, from_lerobot(right, G_LO, G_HI), atol=0.01)


def test_virtual_leaders_expose_lerobot_teleoperator_surface(setup):
    robot, ctrl, clock = setup
    leader = TrajectoryLeader.hold(HOME_LR, clock)
    assert set(leader.action_features) == set(LEROBOT_FEATURE_NAMES)
    assert leader.feedback_features == {} and leader.is_calibrated and not leader.is_connected
    with leader:  # context manager connects/disconnects, like lerobot Teleoperator
        assert leader.is_connected
        leader.calibrate(); leader.configure(); leader.send_feedback({})
        assert set(leader.get_action()) == set(LEROBOT_FEATURE_NAMES)
    assert not leader.is_connected


def test_lerobot_style_leader_duck_types(setup):
    """Anything with LeRobot's get_action() dict works, e.g. lerobot SO101Leader(use_degrees=True)."""
    robot, ctrl, clock = setup

    class FakeSO101Leader:  # same surface as lerobot.teleoperators.so_leader.SO101Leader
        def connect(self, calibrate: bool = True): ...
        def disconnect(self): ...
        def get_action(self):
            return dict(zip(LEROBOT_FEATURE_NAMES, HOME_LR + 5.0, strict=True))

    sel = CommandSelector(ctrl, "main", {"real_leader": FakeSO101Leader()})
    sel.select("real_leader"); sel.run(2.0)
    np.testing.assert_allclose(robot.get_state().arms["main"].positions,
                               from_lerobot(HOME_LR + 5.0, G_LO, G_HI), atol=0.01)


def test_range_m100_100_leader_drives_follower(setup):
    """A LeRobot leader on its default normalization (use_degrees=False) maps correctly."""
    from so101_vla.conventions import LeRobotUnits
    robot, ctrl, clock = setup
    lim = robot.joint_limits("main")
    units = LeRobotUnits(lim.lower, lim.upper, "range_m100_100")
    target_q = from_lerobot(HOME_LR + np.array([10.0, 5.0, -5.0, 8.0, 20.0, 20.0]), G_LO, G_HI)
    leader = TrajectoryLeader.hold(units.to_lerobot(target_q), clock); leader.connect()
    sel = CommandSelector(ctrl, "main", {"leader": leader}, norm="range_m100_100")
    sel.select("leader"); sel.run(3.0)
    np.testing.assert_allclose(robot.get_state().arms["main"].positions, target_q, atol=0.01)
    obs = robot.get_observation(norm="range_m100_100")
    np.testing.assert_allclose([obs[k] for k in LEROBOT_FEATURE_NAMES], units.to_lerobot(target_q), atol=0.6)
