#!/usr/bin/env python3
"""Small solver-free integration test for the finalized Mondrian core.

This test uses a synthetic smooth table. It does not replace the required
fresh-HJ-solve training smoke test; it only exercises regional calibration,
reachable-region buffer selection, and CP rollout wiring without JAX or ODP.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import conformal_shield as cs


def main() -> None:
    world = cs.World()
    bounds = cs.Interval(-1.0, 1.0)
    variant = cs.DynamicsVariant(
        name="SyntheticMismatch",
        model=cs.DynamicsSpec(v=0.60, beta_u=0.80, v_min=-0.20),
        truth=cs.DynamicsSpec(v=0.60, beta_u=0.60, v_min=0.0),
    )
    x_grid = np.linspace(-4.0, 4.0, 11)
    y_grid = np.linspace(-4.0, 4.0, 11)
    th_grid = np.linspace(-math.pi, math.pi, 9)
    tau = np.linspace(-1.0, 0.0, 11)
    X, Y, TH = np.meshgrid(x_grid, y_grid, th_grid, indexing="ij")
    signed_distance = np.sqrt(X ** 2 + Y ** 2) - 0.70
    smooth_value = np.clip(signed_distance, -1.0, 1.0) + 0.05 * np.cos(TH)
    table = cs.CBVFTable(
        x_grid=x_grid,
        y_grid=y_grid,
        th_grid=th_grid,
        tau=tau,
        B_table=np.repeat(smooth_value[..., None], len(tau), axis=3),
        max_abs_table=2.0,
        max_abs_grad=20.0,
    )
    partition = cs.MondrianPartition(
        clearance_edges=(0.25, 0.75, 1.50),
        obstacle_center=world.obstacle_center,
        inflated_obstacle_radius=world.obstacle_radius + world.robot_radius,
    )
    rng = np.random.default_rng(123)
    initial_states = [
        np.array(
            [
                -2.0 + rng.uniform(-0.20, 0.20),
                -2.0 + rng.uniform(-0.20, 0.20),
                math.pi / 4.0 + rng.uniform(-0.10, 0.10),
            ],
            dtype=float,
        )
        for _ in range(24)
    ]
    shield_cfg = cs.ShieldConfig(
        cbvf_activate_margin=0.0,
        cp_activate_margin=0.10,
    )

    calibration, stats = cs.calibrate_mondrian(
        world=world,
        control_bounds=bounds,
        variant=variant,
        cbvf=table,
        policy=lambda state: np.array([-1.0, 0.0]),
        tau=tau,
        init_states=initial_states,
        gamma=0.10,
        delta_traj=0.20,
        shield_cfg=shield_cfg,
        continue_after_unsafe=True,
        partition=partition,
        epsilon_grid=0.0,
    )
    assert calibration.n_effective_regions >= 1
    assert math.fsum(calibration.deltas) <= 0.20
    assert all(value >= 0.0 for value in calibration.buffers)
    assert calibration.max_buffer > 0.0

    env = cs.DubinsCBVFEnv(
        world=world,
        control_bounds=bounds,
        variant=variant,
        dt=0.10,
        horizon=10,
    )
    result = cs.rollout(
        env=env,
        cbvf=table,
        policy=lambda state: np.array([-1.0, 0.0]),
        tau=tau,
        init_state=initial_states[0],
        method="cp",
        gamma=0.10,
        xi=0.0,
        shield_cfg=shield_cfg,
        mondrian_calibration=calibration,
        score_partition=partition,
        speed_envelope=0.95,
        epsilon_inter=0.0,
    )
    assert result["steps"] > 0
    assert result["buffer_coverage_available"] == 1.0
    assert result["max_candidate_effective_regions"] >= 1.0
    print("Synthetic Mondrian integration smoke test: PASS")
    print(
        f"M_effective={calibration.n_effective_regions}, "
        f"xi_max={calibration.max_buffer:.6f}, "
        f"n_calib={int(stats['n_calib'])}"
    )


if __name__ == "__main__":
    main()
