#!/usr/bin/env python3
"""Solver-free checks for feasible projection and infeasible fallback semantics."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import conformal_shield as cs


class FakeCBVF:
    def __init__(self, grad_theta: float, base_without_control: float, gamma: float,
                 grad_xy=(0.0, 0.0)):
        self.B = 0.05
        self.grad_theta = float(grad_theta)
        self.grad_xy = tuple(float(x) for x in grad_xy)
        self.DtB = float(base_without_control - gamma * self.B)

    def value_grad_dt(self, state, time):
        _ = state, time
        return (
            self.B,
            np.array([self.grad_xy[0], self.grad_xy[1], self.grad_theta], dtype=float),
            self.DtB,
        )


def main() -> None:
    gamma = 0.1
    bounds = cs.Interval(lo=-1.0, hi=1.0)
    variant = cs.DynamicsVariant(
        name="projection_smoke",
        model=cs.DynamicsSpec(v=0.6, beta_u=0.5),
        truth=cs.DynamicsSpec(v=0.6, beta_u=0.5),
    )
    env = cs.DubinsCBVFEnv(
        world=cs.World(),
        control_bounds=bounds,
        variant=variant,
        dt=0.05,
        horizon=1,
    )
    state = np.array([-2.0, -2.0, math.pi / 4.0], dtype=float)

    # Genuine two-input case. With symmetric speed bounds, grad_xy aligned to
    # heading gives a_u=[1,1] and zero drift. Projecting [-.5,-.5] onto
    # u_speed+u_yaw>=.5 therefore gives the unique solution [.25,.25].
    two_input_variant = cs.DynamicsVariant(
        name="two_input_projection_smoke",
        model=cs.DynamicsSpec(v=0.6, v_min=-0.6, beta_u=0.5),
        truth=cs.DynamicsSpec(v=0.6, v_min=-0.6, beta_u=0.5),
    )
    two_input_env = cs.DubinsCBVFEnv(
        world=cs.World(), control_bounds=bounds, variant=two_input_variant,
        dt=0.05, horizon=1,
    )
    heading = np.array([math.cos(state[2]), math.sin(state[2])])
    two_input_cbvf = FakeCBVF(
        grad_theta=2.0, base_without_control=0.0, gamma=gamma,
        grad_xy=heading / 0.6,
    )
    two_input = cs.solve_cbvf_projection_exact(
        s=state, t=0.0, u_nom=np.array([-0.5, -0.5]), cbvf=two_input_cbvf,
        env=two_input_env, gamma=gamma, xi=0.5,
    )
    assert two_input.feasible
    assert np.allclose(two_input.a_u, [1.0, 1.0], atol=1e-10)
    assert np.allclose(two_input.u, [0.25, 0.25], atol=1e-9)

    # a_u=1 and base=0, so u>=0.25. The minimum-distance action from -0.5 is 0.25.
    controllable_cbvf = FakeCBVF(
        grad_theta=2.0,
        base_without_control=0.0,
        gamma=gamma,
    )
    feasible = cs.solve_cbvf_projection_exact(
        s=state,
        t=0.0,
        u_nom=np.array([-0.2, -0.5]),
        cbvf=controllable_cbvf,
        env=env,
        gamma=gamma,
        xi=0.25,
    )
    assert feasible.feasible
    assert np.allclose(feasible.u, [-0.2, 0.25], rtol=0.0, atol=1e-10)
    assert feasible.margin >= -1e-10

    # The requested RHS exceeds max_u Psi=1. The documented fallback is the
    # least-violation affine maximizer, which is the upper actuator rail.
    infeasible = cs.solve_cbvf_projection_exact(
        s=state,
        t=0.0,
        u_nom=np.array([-0.2, -0.5]),
        cbvf=controllable_cbvf,
        env=env,
        gamma=gamma,
        xi=1.25,
    )
    assert not infeasible.feasible
    assert infeasible.assumption5_violation
    assert math.isclose(infeasible.max_achievable_lhs, 1.0, abs_tol=1e-10)
    assert np.allclose(infeasible.u, [bounds.hi, bounds.hi], atol=1e-10)
    assert math.isclose(
        infeasible.max_achievable_lhs - infeasible.requested_xi,
        -0.25,
        abs_tol=1e-10,
    )

    # With a_u=0 the actuator cannot improve Psi, so the solver keeps nominal
    # while still reporting the original requested constraint as infeasible.
    no_authority_cbvf = FakeCBVF(
        grad_theta=0.0,
        base_without_control=0.1,
        gamma=gamma,
    )
    no_authority = cs.solve_cbvf_projection_exact(
        s=state,
        t=0.0,
        u_nom=np.array([-0.4, 0.3]),
        cbvf=no_authority_cbvf,
        env=env,
        gamma=gamma,
        xi=0.2,
    )
    assert not no_authority.feasible
    assert no_authority.assumption5_violation
    assert np.allclose(no_authority.u, [-0.4, 0.3], atol=1e-10)

    print("QP projection/fallback smoke test: PASS")
    print("two-input minimal projection u=[0.25, 0.25]")
    print("feasible minimal projection u=0.25")
    print("infeasible affine fallback u=1.0, feasibility_margin=-0.25")
    print("zero-control-authority fallback keeps u_nom=-0.4")


if __name__ == "__main__":
    main()
