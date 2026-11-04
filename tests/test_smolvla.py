"""Real SmolVLA checkpoint (Task 2.3/2.5). Skipped without the vla extra, CUDA, or the downloaded model.

Run ``uv run python scripts/m5_offline_inference.py`` once to download the checkpoint.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lerobot")
hub = pytest.importorskip("huggingface_hub")

from so101_vla.config import load_robot_config  # noqa: E402
from so101_vla.conventions import LeRobotUnits  # noqa: E402
from so101_vla.policies import ActionChunkValidator  # noqa: E402
from so101_vla.policies.smolvla import SmolVLAConfig, SmolVLAPolicy  # noqa: E402
from so101_vla.robot.mujoco_robot import MuJoCoRobot  # noqa: E402
from so101_vla.sim.tabletop import PickPlaceTask  # noqa: E402

CFG = SmolVLAConfig()
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)
if not isinstance(hub.try_to_load_from_cache(CFG.checkpoint, "model.safetensors", revision=CFG.revision), str):
    pytest.skip("smolvla_base not downloaded yet", allow_module_level=True)


@pytest.fixture(scope="module")
def scene():
    cfg = load_robot_config("configs/robot/so101_tabletop.yaml")
    robot = MuJoCoRobot.from_config(cfg)
    PickPlaceTask(robot).reset(np.asarray(cfg.poses["home"]), seed=0)
    yield robot
    robot.close()


@pytest.mark.parametrize("mode", ["as_shipped", "checkpoint_action_stats"])
def test_checkpoint_contract_and_one_chunk(scene, mode):
    policy = SmolVLAPolicy(SmolVLAConfig(stats_mode=mode))
    info = policy.describe()
    assert info["input_features"]["observation.state"]["shape"] == [6]
    assert set(info["input_features"]) == {"observation.state", *(f"observation.images.camera{i}" for i in (1, 2, 3))}
    assert info["output_features"]["action"]["shape"] == [6] and info["chunk_size"] == 50
    # Finding: saved stats load under legacy keys, so nothing applies unless overridden.
    loaded = info["normalizer_stats_keys_loaded"]["preprocessor.NormalizerProcessorStep"]
    if mode == "as_shipped":
        assert "action" not in loaded and "observation.state" not in loaded
    else:
        assert {"action", "observation.state"} <= set(loaded)
    obs = scene.get_observation(cameras=policy.camera_names, width=256, height=256)
    chunk = policy.predict_chunk(obs, "Pick up the cube.")
    lim = scene.joint_limits("main")
    report = ActionChunkValidator(LeRobotUnits(lim.lower, lim.upper)).validate(chunk, scene.get_state().arms["main"].positions)
    assert report.steps == 50 and report.finite  # Level 1: valid action tensors


def test_rtc_guided_second_chunk(scene):
    """LeRobot RTC: the second chunk is guided by the unexecuted tail of the first (no grad errors)."""
    policy = SmolVLAPolicy(SmolVLAConfig(rtc=True))
    assert policy.uses_rtc and policy.rtc_replan_every == 20
    obs = scene.get_observation(cameras=policy.camera_names, width=256, height=256)
    first = policy.predict_chunk(obs, "Pick up the cube.")
    second = policy.predict_chunk(obs, "Pick up the cube.", executed_from_previous=20)
    assert first.shape == second.shape == (50, 6)
    assert np.all(np.isfinite(second))
