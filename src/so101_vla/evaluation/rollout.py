"""Closed-loop VLA episodes on the tabletop task, with the three evaluation levels (Tasks 2.6, 2.7).

Loop (design.md Task 2.6, with the real APIs):

    reset task (seeded) -> CommandSelector selects the PolicySource
    every control tick:   PolicySource.get_action()  (re-observes + predicts when its chunk runs out)
                          -> LeRobot units -> rad -> SafetyFilter -> MuJoCo actuators -> physics substeps
    stop on success, on a source error (the selector then holds), or at the episode timeout

Levels are reported separately and only from what was measured:

    level1_inference:  every predicted chunk had the right shape and finite values
    level2_control:    the predicted actions were mapped and executed with no source error,
                       no backend rejection, and no arm-table collision
    level3_success:    the cube was detected in the box (LeRobot env definition) at the end
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from so101_vla.control.controller import ArmController, TrackingLog
from so101_vla.policies.policy_source import PolicySource
from so101_vla.robot.mujoco_robot import MuJoCoRobot
from so101_vla.sim.scene import arm_prefix
from so101_vla.sim.tabletop import PickPlaceTask
from so101_vla.teleop.selector import CommandSelector
from so101_vla.utils.images import save_png


@dataclass(frozen=True)
class EpisodeConfig:
    instruction: str
    seed: int
    max_seconds: float = 30.0
    start_pose: str = "home"
    success_hold_s: float = 0.5  # in the box for this long counts as done


def _arm_table_contacts(robot: MuJoCoRobot, arm: str) -> int:
    m, d = robot.model, robot.data
    table = m.geom("table").id
    prefix = arm_prefix(arm)
    n = 0
    for i in range(d.ncon):
        g1, g2 = int(d.contact[i].geom1), int(d.contact[i].geom2)
        other = g2 if g1 == table else g1 if g2 == table else None
        if other is not None and m.body(int(m.geom_bodyid[other])).name.startswith(prefix):
            n += 1
    return n


def run_episode(robot: MuJoCoRobot, ctrl: ArmController, task: PickPlaceTask, source: PolicySource,
                ep: EpisodeConfig, out_dir: Path | None = None, arm: str = "main") -> dict:
    task.reset(np.asarray(ctrl.resolve_pose(ep.start_pose)[arm]), seed=ep.seed)
    ctrl.sync()
    ctrl.log = TrackingLog(robot.arm_names)  # per-episode tracking log
    source.reset(ep.instruction)
    sel = CommandSelector(ctrl, arm, {"policy": source})
    sel.select("policy")

    ticks = int(round(ep.max_seconds / ctrl.dt))
    t_wall = time.perf_counter()
    states, table_hits, in_box_since, stop = [], 0, None, "timeout"
    min_dist, touched, grasped, lifted = np.inf, False, False, False
    for _ in range(ticks):
        sel.tick()
        s = task.state()
        states.append((s.time, *s.cube_pos, *s.gripper_pos, s.gripper_to_cube, s.touching, s.grasped, s.lifted, s.in_box))
        hits = _arm_table_contacts(robot, arm)
        table_hits += hits > 0
        min_dist = min(min_dist, s.gripper_to_cube)
        touched, grasped, lifted = touched or s.touching, grasped or s.grasped, lifted or s.lifted
        in_box_since = (in_box_since if in_box_since is not None else s.time) if s.in_box else None
        if in_box_since is not None and s.time - in_box_since >= ep.success_hold_s:
            stop = "success"
            break
        if sel.active is None:
            stop = "source_error"
            break
    final = task.state()
    errors = [e for e in sel.log.events if "source_error" in e]
    reports = [c.report for c in source.chunks]
    tl = ctrl.log.summary().get(arm, {})
    result = {
        "instruction": ep.instruction, "seed": ep.seed, "layout": task.layout, "stop": stop,
        "sim_seconds": round(final.time, 3), "wall_seconds": round(time.perf_counter() - t_wall, 1),
        "chunks": len(source.chunks), "actions_executed": len(source.executed),
        "inference_ms_mean": round(float(np.nanmean([c.inference_ms or np.nan for c in source.chunks])), 1)
        if source.chunks else None,
        "level1_inference": bool(source.chunks) and not any("InvalidChunk" in e["error"] for e in errors),
        "level2_control": not errors and table_hits == 0,
        "level3_success": stop == "success",
        "source_errors": errors,
        "chunks_with_out_of_range_actions": sum(r.any_out_of_range for r in reports),
        "out_of_range_action_steps": {j: int(sum(r.out_of_range_steps[j] for r in reports))
                                      for j in reports[0].out_of_range_steps} if reports else {},
        "ticks_position_clipped": tl.get("ticks_position_clipped"),
        "ticks_rate_limited": tl.get("ticks_rate_limited"),
        "ticks_arm_touching_table": int(table_hits),
        "progress": {"min_gripper_to_cube_m": round(float(min_dist), 4), "touched_cube": touched,
                     "grasped_cube": grasped, "lifted_cube": lifted, "cube_in_box_at_end": final.in_box,
                     "cube_moved_m": round(float(np.linalg.norm(final.cube_pos[:2] - np.array(task.layout["cube_xy"]))), 4)},
    }
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / "policy.npz", **source.arrays())
        np.savez_compressed(out_dir / "task_state.npz", state=np.array(states, float), columns=np.array(
            ["time", "cube_x", "cube_y", "cube_z", "grip_x", "grip_y", "grip_z", "gripper_to_cube",
             "touching", "grasped", "lifted", "in_box"]))
        sel.log.save(out_dir, ctrl.dt)
        ctrl.log.save(out_dir)
        result["chunk_reports"] = [r.to_dict() for r in reports]
        (out_dir / "episode.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        for cam in source.policy.camera_names:  # first and last observation the policy saw
            if source.chunks:
                save_png(out_dir / f"first_{cam.replace('/', '_')}.png", source.chunks[0].frames[cam])
                save_png(out_dir / f"last_{cam.replace('/', '_')}.png", source.chunks[-1].frames[cam])
        result.pop("chunk_reports")
    return result
