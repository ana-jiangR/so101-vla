import numpy as np
import pytest

from so101_vla.config import SafetyConfig
from so101_vla.control.safety import SafetyFilter, SafetyLimits
from so101_vla.control.trajectory import min_jerk, min_jerk_duration
from so101_vla.robot.interface import CommandError, JointLimits

DT = 0.02
HARD = JointLimits(lower=np.full(6, -1.5), upper=np.full(6, 1.5))
CFG = SafetyConfig(position_margin=0.05, max_velocity=(1.0,) * 6, max_acceleration=(4.0,) * 6)


def make_filter(mode: str = "clip") -> SafetyFilter:
    f = SafetyFilter(SafetyLimits.from_config(HARD, CFG), DT, mode)
    f.reset(np.zeros(6))
    return f


@pytest.mark.parametrize("bad", [np.zeros(5), np.zeros((2, 6)), np.array([0, 0, np.nan, 0, 0, 0]),
                                 np.array([0, 0, 0, np.inf, 0, 0])])
def test_malformed_targets_raise(bad):
    with pytest.raises(CommandError):
        make_filter()(bad)


def test_out_of_range_clip_vs_reject():
    target = np.array([5.0, 0, 0, 0, 0, -5.0])
    f = make_filter("clip")
    r = f(target)
    assert r.position_clipped.tolist() == [True, False, False, False, False, True]
    goal, _ = f.check_range(target)
    assert goal[0] == pytest.approx(1.45) and goal[5] == pytest.approx(-1.45)
    with pytest.raises(CommandError, match="soft limits"):
        make_filter("reject")(target)


def test_step_command_respects_velocity_and_acceleration_and_does_not_overshoot():
    f = make_filter()
    goal = np.array([1.0, -0.5, 0, 0, 0, 0.3])
    qs = [np.zeros(6)]
    for _ in range(300):
        qs.append(f(goal).target)
    qs = np.array(qs)
    vel = np.diff(qs, axis=0) / DT
    acc = np.diff(vel, axis=0) / DT
    assert np.abs(vel).max() <= 1.0 + 1e-9
    assert np.abs(acc).max() <= 4.0 + 1e-6
    np.testing.assert_allclose(qs[-1], goal, atol=1e-9)
    # monotone approach: never past the goal
    assert np.all(qs[:, 0] <= goal[0] + 1e-9) and np.all(qs[:, 1] >= goal[1] - 1e-9)


def test_smooth_trajectory_within_limits_passes_unchanged():
    f = make_filter()
    q1 = np.array([1.0, -0.8, 0.5, 0.2, -0.3, 0.6])
    T = min_jerk_duration(np.zeros(6), q1, np.full(6, 1.0), np.full(6, 4.0))
    traj = min_jerk(np.zeros(6), q1, T, DT)
    for k in range(1, len(traj.times)):
        r = f(traj.positions[k])
        assert not r.rate_limited.any(), f"limited at tick {k}"
        np.testing.assert_allclose(r.target, traj.positions[k], atol=1e-12)


def test_max_relative_target_bounds_target_to_measured_position():
    # A blocked joint (measured stuck at 0) must not let the target run away,
    # as LeRobot's ensure_safe_goal_position on the real follower.
    cfg = SafetyConfig(position_margin=(0.05,) * 6, max_velocity=(1.0,) * 6, max_acceleration=(4.0,) * 6,
                       max_relative_target=(0.1,) * 6)
    f = SafetyFilter(SafetyLimits.from_config(HARD, cfg), DT)
    f.reset(np.zeros(6))
    stuck = np.zeros(6)
    for _ in range(200):
        r = f(np.full(6, 1.0), measured=stuck)
    assert np.all(r.target <= 0.1 + 1e-12)
    assert r.relative_limited.all() and r.modified
    # once the joint is free and catches up, the target proceeds to the goal
    for _ in range(300):
        r = f(np.full(6, 1.0), measured=f.last_target)
    np.testing.assert_allclose(r.target, 1.0, atol=1e-9)
    # off by default: no measured-based limiting
    assert SafetyLimits.from_config(HARD, CFG).max_relative_target is None


def test_output_never_leaves_soft_limits():
    f = make_filter()
    rng = np.random.default_rng(0)
    for _ in range(500):
        q = f(rng.uniform(-3, 3, 6)).target
        assert np.all(q >= -1.45 - 1e-12) and np.all(q <= 1.45 + 1e-12)
