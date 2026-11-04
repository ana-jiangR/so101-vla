import numpy as np
import pytest

from so101_vla.control.trajectory import Trajectory, min_jerk, min_jerk_duration, waypoint_trajectory
from so101_vla.conventions import (
    LEROBOT_FEATURE_NAMES,
    from_lerobot,
    from_lerobot_dict,
    gripper_to_normalized,
    normalized_to_gripper,
    to_lerobot,
    to_lerobot_dict,
)

# Menagerie SO-101 gripper ctrlrange
G_LO, G_HI = -0.17453, 1.74533


def test_lerobot_units_match_official_env_mapping():
    # LeRobot env HOME (deg x5, gripper 0..100) <-> radians
    home_lr = np.array([0.0, -60.0, 60.0, 35.0, 0.0, 30.0])
    q = from_lerobot(home_lr, G_LO, G_HI)
    np.testing.assert_allclose(q[:5], np.deg2rad(home_lr[:5]))
    assert q[5] == pytest.approx(G_LO + 0.3 * (G_HI - G_LO))
    np.testing.assert_allclose(to_lerobot(q, G_LO, G_HI), home_lr, atol=1e-9)
    # gripper endpoints: 0 = closed (lower limit), 100 = open (upper limit)
    assert to_lerobot(np.array([0, 0, 0, 0, 0, G_LO]), G_LO, G_HI)[5] == pytest.approx(0.0)
    assert to_lerobot(np.array([0, 0, 0, 0, 0, G_HI]), G_LO, G_HI)[5] == pytest.approx(100.0)


def test_lerobot_dict_roundtrip_and_missing_keys():
    q = np.array([0.1, -0.2, 0.3, -0.4, 0.5, 0.6])
    d = to_lerobot_dict(q, G_LO, G_HI)
    assert tuple(d) == LEROBOT_FEATURE_NAMES and "shoulder_pan.pos" in d
    np.testing.assert_allclose(from_lerobot_dict(d, G_LO, G_HI), q)
    del d["gripper.pos"]
    with pytest.raises(KeyError, match="gripper.pos"):
        from_lerobot_dict(d, G_LO, G_HI)

DT = 0.02


def test_min_jerk_endpoints_and_limits():
    q0, q1 = np.zeros(6), np.array([1.0, -1.0, 0.5, 0, 0, 0.2])
    vmax, amax = np.full(6, 1.5), np.full(6, 6.0)
    T = min_jerk_duration(q0, q1, vmax, amax)
    traj = min_jerk(q0, q1, T, 0.001)
    np.testing.assert_allclose(traj.start, q0)
    np.testing.assert_allclose(traj.end, q1)
    vel = np.diff(traj.positions, axis=0) / 0.001
    assert np.abs(vel).max() <= 1.5 * 1.001
    acc = np.diff(vel, axis=0) / 0.001
    assert np.abs(acc).max() <= 6.0 * 1.01


def test_waypoints_visit_each_point():
    wps = [np.full(6, 0.3), np.full(6, -0.2), np.zeros(6)]
    traj = waypoint_trajectory(np.zeros(6), wps, np.full(6, 1.0), np.full(6, 4.0), DT)
    for wp in wps:
        assert np.min(np.abs(traj.positions - wp).max(axis=1)) < 1e-9


def test_trajectory_validation_and_concat():
    with pytest.raises(ValueError):
        Trajectory(np.array([0.0, 0.0]), np.zeros((2, 6)))
    a = min_jerk(np.zeros(6), np.ones(6), 1.0, DT)
    with pytest.raises(ValueError, match="continuous"):
        a.then(min_jerk(np.zeros(6), np.ones(6), 1.0, DT))
    ab = a.then(min_jerk(np.ones(6), np.zeros(6), 0.5, DT))
    assert ab.duration == pytest.approx(1.5)


def test_gripper_normalization_roundtrip():
    lo, hi = -0.17453, 1.74533
    assert normalized_to_gripper(0.0, lo, hi) == pytest.approx(lo)
    assert normalized_to_gripper(1.0, lo, hi) == pytest.approx(hi)
    for x in (0.0, 0.25, 0.9):
        assert gripper_to_normalized(normalized_to_gripper(x, lo, hi), lo, hi) == pytest.approx(x)
    with pytest.raises(ValueError):
        normalized_to_gripper(1.2, lo, hi)


# Menagerie SO-101 hard limits (docs/conventions.md)
LO = np.array([-1.91986, -1.74533, -1.69, -1.65806, -2.74385, G_LO])
HI = np.array([1.91986, 1.74533, 1.69, 1.65806, 2.84121, G_HI])


def test_lerobot_units_degrees_mode_equals_official_env_functions():
    from so101_vla.conventions import LeRobotUnits
    u = LeRobotUnits(LO, HI)  # default "degrees"
    q = np.array([0.1, -0.2, 0.3, -0.4, 0.5, 0.6])
    np.testing.assert_allclose(u.to_lerobot(q), to_lerobot(q, G_LO, G_HI))
    np.testing.assert_allclose(u.from_dict(u.to_dict(q, "left_"), "left_"), q)


def test_lerobot_units_range_mode_matches_lerobot_normalization():
    """range_m100_100: limits -> -100/+100 linearly (LeRobot's default, use_degrees=False)."""
    from so101_vla.conventions import LeRobotUnits
    u = LeRobotUnits(LO, HI, "range_m100_100")
    np.testing.assert_allclose(u.to_lerobot(LO), [-100, -100, -100, -100, -100, 0], atol=1e-9)
    np.testing.assert_allclose(u.to_lerobot(HI), [100, 100, 100, 100, 100, 100], atol=1e-9)
    mid = (LO + HI) / 2
    np.testing.assert_allclose(u.to_lerobot(mid)[:5], 0.0, atol=1e-9)
    q = np.array([0.1, -0.2, 0.3, -0.4, 0.5, 0.6])
    np.testing.assert_allclose(u.from_lerobot(u.to_lerobot(q)), q, atol=1e-12)
    with pytest.raises(ValueError, match="norm mode"):
        LeRobotUnits(LO, HI, "radians")
    with pytest.raises(ValueError, match="shape"):
        LeRobotUnits(LO[:5], HI[:5])
