#!/usr/bin/env python3
"""Deterministic smoke checks for the continuous inter-sample implementation."""

import math
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import conformal_shield as cs
from redexp.envs.conformal_dubins_3d_env import ConformalDubins3dEnv


def main() -> None:
    partition = cs.MondrianPartition(
        clearance_edges=(0.25, 0.75, 1.5),
        obstacle_center=(0.0, 0.0),
        inflated_obstacle_radius=0.7,
    )
    calibration = cs.MondrianCalibration(
        partition=partition,
        effective_groups=((0,), (1,), (2,), (3,)),
        base_to_effective=(0, 1, 2, 3),
        buffers=(0.1, 0.2, 0.3, 0.4),
        deltas=(0.0125, 0.0125, 0.0125, 0.0125),
        counts=(100, 100, 100, 100),
        delta_traj=0.05,
        epsilon_grid=0.0,
        promotion_rounds=0,
    )
    state = np.array([0.94, 0.0, 0.0])
    applied, reachable = calibration.applied_buffer(state, travel_radius=0.02)
    assert reachable == (0, 1), reachable
    assert applied == 0.2, applied

    x = np.linspace(-1.0, 1.0, 5)
    y = np.linspace(-1.0, 1.0, 5)
    th = np.linspace(-math.pi, math.pi, 7, endpoint=False)
    tau = np.linspace(-0.2, 0.0, 3)
    X, Y, TH, T = np.meshgrid(x, y, th, tau, indexing="ij")
    table = X + 2.0 * Y + 0.1 * np.cos(TH) + 0.5 * T
    cbvf = cs.CBVFTable(x, y, th, tau, table, max_abs_table=10.0)
    cbvf.enable_continuous_time_interpolation(True)
    certificate = cbvf.certified_global_psi_lipschitz(
        deployment_tau=np.linspace(-0.2, 0.0, 5),
        model_spec=cs.DynamicsSpec(v=0.6, v_min=-0.2, beta_u=0.8),
        gamma=0.1,
    )
    values = np.asarray([row["L_Psi_j"] for row in certificate["intervals"]])
    assert np.isfinite(values).all() and np.all(values >= 0.0)
    local = cbvf.certified_local_psi_lipschitz(
        state=np.array([0.0, 0.0, 0.0]),
        t_start=-0.2, t_end=-0.15,
        position_radius=0.6 * 0.05,
        heading_radius=0.8 * 0.05,
        model_spec=cs.DynamicsSpec(v=0.6, v_min=-0.2, beta_u=0.8),
        gamma=0.1,
    )
    assert np.isfinite(local["L_Psi_j"]) and local["L_Psi_j"] >= 0.0
    assert local["L_Psi_j"] <= float(np.max(values)) + 1e-12
    assert local["method"] == "reachable_tube_local_piecewise_multilinear_bound"
    local_eta = cbvf.certified_local_eta_time_lipschitz(
        state=np.array([0.0, 0.0, 0.0]), t_start=-0.2, t_end=-0.15,
        position_radius=0.03, heading_radius=0.04,
        model_spec=cs.DynamicsSpec(v=0.6, v_min=-0.2, beta_u=0.8),
        truth_spec=cs.DynamicsSpec(v=0.6, v_min=0.0, beta_u=0.6),
    )
    assert local_eta["state_lipschitz_bound"] >= 0.0
    assert local_eta["time_lipschitz_bound"] >= 0.0
    C_x = math.hypot(0.6, 0.8)
    epsilon = values * (C_x + 1.0) * 0.05
    assert np.allclose(epsilon, values * 0.1)

    env = ConformalDubins3dEnv(
        initial_state=(0.701, 0.0, math.pi), random_start=False,
        speed=0.6, speed_min=0.6, beta_u=0.0, dt=0.05,
        integration_substeps=10, terminate_on_unsafe=True,
    )
    env.reset(seed=0)
    _, _, terminated, _, info = env.step(np.array([1.0, 0.0]))
    assert terminated
    assert info["intersample_unsafe"]
    assert info["minimum_substep_safe_margin"] < 0.0
    assert info["controller_update_safe_margin"] > 0.0
    print("Inter-sample protection smoke test: PASS")


if __name__ == "__main__":
    main()
