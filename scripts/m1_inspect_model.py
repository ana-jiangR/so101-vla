"""Milestone 1: print the compiled MuJoCo model's joints, actuators, and conventions.

Reads the *compiled* model (after defaults/classes are resolved), which is the
source of truth for what data.ctrl means. Findings are recorded in
docs/conventions.md.

Usage:
    uv run python scripts/m1_inspect_model.py [path/to/model.xml]
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "assets" / "robotstudio_so101" / "so101.xml"

np.set_printoptions(precision=4, suppress=True)


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MODEL
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    print(f"model: {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    print(f"timestep={m.opt.timestep}s  integrator={mujoco.mjtIntegrator(m.opt.integrator).name}")
    print(f"nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody} ngeom={m.ngeom} ncam={m.ncam} nsite={m.nsite}")

    print("\nJOINTS (qpos order)")
    print(f"{'id':>2} {'name':<14} {'type':<6} {'qadr':>4} {'range [rad]':<20} {'range [deg]':<18} "
          f"{'damping':>7} {'friction':>8} {'armature':>8}")
    for j in range(m.njnt):
        jt = mujoco.mjtJoint(m.jnt_type[j]).name.removeprefix("mjJNT_").lower()
        lo, hi = m.jnt_range[j]
        dof = m.jnt_dofadr[j]
        print(f"{j:>2} {m.joint(j).name:<14} {jt:<6} {m.jnt_qposadr[j]:>4} "
              f"[{lo:+.4f}, {hi:+.4f}]   [{np.degrees(lo):+7.2f}, {np.degrees(hi):+7.2f}] "
              f"{m.dof_damping[dof]:>7.3f} {m.dof_frictionloss[dof]:>8.3f} {m.dof_armature[dof]:>8.3f}")

    print("\nACTUATORS (ctrl order)")
    for a in range(m.nu):
        trn = mujoco.mjtTrn(m.actuator_trntype[a]).name
        target = m.joint(m.actuator_trnid[a, 0]).name
        gain = mujoco.mjtGain(m.actuator_gaintype[a]).name
        bias = mujoco.mjtBias(m.actuator_biastype[a]).name
        kp = m.actuator_gainprm[a, 0]
        # Position actuator: force = kp*ctrl - kp*qpos - kv*qvel  ->  biasprm = [0, -kp, -kv]
        b0, b1, b2 = m.actuator_biasprm[a, :3]
        print(f"{a:>2} {m.actuator(a).name:<14} -> {trn}:{target:<14} gear={m.actuator_gear[a, 0]:g} "
              f"{gain}/{bias} kp={kp:.3f} bias=[{b0:g}, {b1:.3f}, {b2:.3f}]")
        print(f"   ctrlrange={m.actuator_ctrlrange[a]} (limited={bool(m.actuator_ctrllimited[a])})  "
              f"forcerange={m.actuator_forcerange[a]} (limited={bool(m.actuator_forcelimited[a])})")

    print("\nSITES")
    for s in range(m.nsite):
        print(f"  {m.site(s).name:<14} body={m.body(m.site_bodyid[s]).name:<22} world_pos={d.site_xpos[s]}")
    print("CAMERAS")
    for c in range(m.ncam):
        print(f"  {m.camera(c).name:<14} body={m.body(m.cam_bodyid[c]).name} fovy={m.cam_fovy[c]}")

    print(f"\nINITIAL STATE  qpos0={m.qpos0}")
    print(f"keyframes: {[m.key(k).name for k in range(m.nkey)]}")

    # Contacts at the zero pose (self-collision sanity check).
    print(f"contacts at qpos0: {d.ncon}")
    for i in range(d.ncon):
        c = d.contact[i]
        print(f"  {m.body(m.geom_bodyid[c.geom1]).name} <-> {m.body(m.geom_bodyid[c.geom2]).name} dist={c.dist:.4f}")

    # Settle with ctrl = qpos0 and report drift: shows whether ctrl == joint target.
    d.ctrl[:] = d.qpos[m.actuator_trnid[:, 0]]
    for _ in range(int(2.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    print(f"\nafter 2s holding ctrl=qpos0: qpos={d.qpos}  |qvel|max={np.abs(d.qvel).max():.2e}  ncon={d.ncon}")


if __name__ == "__main__":
    main()
