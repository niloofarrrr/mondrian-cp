"""
solver_cbvf.py  –  Robust HJ/CBVF backward-time solver for conformal CBVF shielding.

This file keeps the numerically stable solver path from the last working
revision. The outer experiment script now handles the latest experimental
changes: stronger CP diagnostics, anti-livelock release in the CP-only
activation band, and safer preflight grid-size checks.

This solver keeps the earlier fixes and adds numerical guards that match
what the recent runtime logs show:

1. Hamiltonian/flux arrays are never treated like value functions.
2. The maxCBF exponential uses exp(-γ·dt), not exp(+γ·dt).
3. Backend fluxes are sanitized before they update V.
4. Midpoint updates are done in float64 and clipped before casting back.
5. If RK2 becomes unstable on a micro-step, the solver falls back to Euler,
   and then to a conservative hold step instead of crashing.
"""

import time
from typing import Any, Tuple

import heterocl as hcl
import numpy as np
from odp.computeGraphs import graph_1D, graph_2D, graph_3D, graph_4D, graph_5D, graph_6D


def _build_graph(dynamics_obj: Any, grid: Any, target_mode: str, accuracy: str):
    builders = {
        1: graph_1D, 2: graph_2D, 3: graph_3D,
        4: graph_4D, 5: graph_5D, 6: graph_6D,
    }
    if grid.dims not in builders:
        raise ValueError(f"Unsupported grid dimension {grid.dims}. Expected 1 through 6.")
    return builders[grid.dims](dynamics_obj, grid, target_mode, accuracy)


def _build_state_axes(grid: Any) -> Tuple[Any, ...]:
    axes = []
    for dim in range(grid.dims):
        grid_axis = np.reshape(grid.vs[dim], grid.pts_each_dim[dim]).astype(np.float32)
        axes.append(hcl.asarray(grid_axis))
    return tuple(axes)


def _call_graph(solve_pde: Any, grid: Any, hamiltonian: Any, value: Any,
                axes: Tuple[Any, ...], delta_t: Any, t: Any, l0: Any) -> None:
    if grid.dims == 1:
        solve_pde(hamiltonian, value, axes[0], delta_t, t, l0)
    elif grid.dims == 2:
        solve_pde(hamiltonian, value, axes[0], axes[1], delta_t, t, l0)
    elif grid.dims == 3:
        solve_pde(hamiltonian, value, axes[0], axes[1], axes[2], delta_t, t, l0)
    elif grid.dims == 4:
        solve_pde(hamiltonian, value, axes[0], axes[1], axes[2], axes[3], delta_t, t, l0)
    elif grid.dims == 5:
        solve_pde(hamiltonian, value, axes[0], axes[1], axes[2], axes[3], axes[4], delta_t, t, l0)
    elif grid.dims == 6:
        solve_pde(hamiltonian, value, axes[0], axes[1], axes[2], axes[3],
                  axes[4], axes[5], delta_t, t, l0)
    else:
        raise ValueError(f"Unsupported grid dimension {grid.dims}. Expected 1 through 6.")


def _check_value(name: str, arr: np.ndarray, max_abs_value: float) -> None:
    """Guard on the VALUE FUNCTION only.  Never call on Hamiltonian/flux."""
    if not np.isfinite(arr).all():
        raise FloatingPointError(
            f"Non-finite values in value function '{name}'. "
            "Try smaller gamma, coarser grid, or larger cbvf_dt."
        )
    max_abs = float(np.max(np.abs(arr)))
    if max_abs > max_abs_value:
        raise FloatingPointError(
            f"Value magnitude too large in '{name}': "
            f"max|V|={max_abs:.3e} > {max_abs_value:.3e}. "
            "Try smaller target_clip or smaller gamma."
        )


def _numeric_clip_bounds(value_clip_lo, value_clip_hi, max_abs_value: float) -> Tuple[float, float]:
    """Return finite clipping bounds used for intermediate solver states."""
    lo = -float(max_abs_value) if value_clip_lo is None else float(value_clip_lo)
    hi =  float(max_abs_value) if value_clip_hi is None else float(value_clip_hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        raise ValueError(f"Invalid clip bounds: lo={lo}, hi={hi}")
    return lo, hi


def _clip_candidate(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Clip in float64 to avoid float32 overflow during intermediate updates."""
    out = np.asarray(arr, dtype=np.float64)
    out = np.clip(out, lo, hi)
    return out

def _sanitize_flux(arr: np.ndarray, dt: float, max_abs_value: float, factor: float = 8.0) -> np.ndarray:
    """
    Make the backend Hamiltonian numerically safe for one micro-step.

    The ODP backend occasionally emits NaN/Inf spatial flux values on difficult
    micro-steps. Those are not meaningful physical values; they are numerical
    pathologies. Replacing them by zero and clipping the remaining finite fluxes
    prevents the offline solve from crashing while keeping the one-step update
    bounded.
    """
    out = np.asarray(arr, dtype=np.float64)
    if not np.isfinite(out).all():
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    dt_safe = max(float(dt), 1e-12)
    flux_cap = max(float(factor) * float(max_abs_value) / dt_safe, 1.0)
    out = np.clip(out, -flux_cap, flux_cap)
    return out


def _finalize_candidate(
    candidate: np.ndarray,
    *,
    target_mode: str,
    gamma_val: float,
    dt: float,
    l0_step_next: np.ndarray,
    lo: float,
    hi: float,
) -> np.ndarray:
    """Apply the maxCBF clamp and hard bounds in float64."""
    out = np.asarray(candidate, dtype=np.float64)
    if target_mode == "maxCBF":
        out = np.minimum(out * np.exp(-gamma_val * float(dt)), np.asarray(l0_step_next, dtype=np.float64))
    out = np.clip(out, lo, hi)
    return out


def HJSolver(
    dynamics_obj,
    grid,
    multiple_value,
    tau,
    compMethod,
    saveAllTimeSteps=False,
    accuracy="medium",
    untilConvergent=False,
    epsilon=2e-3,
    computeTimeToReach=False,
):
    """
    Customized HJ/CBVF solver for the conformal CBVF shielding experiment.

    tau[0] = 0  (terminal condition, physical time t = 0)
    tau[-1] = T (deepest backward time, physical time t = -T)

    When saveAllTimeSteps=True, the returned array valfuncs has shape
    (*grid_shape, len(tau)) with the original time-reversed convention:
        valfuncs[..., -1]    = terminal slice   (tau[0]  = 0)
        valfuncs[..., -1-i]  = slice at tau[i]
        valfuncs[..., 0]     = deepest backward (tau[-1] = T)

    compMethod keys
    ---------------
    TargetSetMode   : 'maxCBF' for the discounted CBVF formulation
    cbf_gamma       : γ ≥ 0  (default 0)
    value_clip_lo   : lower hard clip on V after each step (float or None)
    value_clip_hi   : upper hard clip on V after each step (float or None)
    max_abs_value   : explosion guard on VALUE FUNCTION ONLY (default 1e4)
    """
    if computeTimeToReach:
        raise NotImplementedError("computeTimeToReach is not used in this customized CBVF solver.")

    print("Welcome to customized optimized_dp CBVF solver\n")

    if isinstance(multiple_value, list):
        target = multiple_value[0]
        constraint = multiple_value[1]
    else:
        target = multiple_value
        constraint = None

    hcl.init()
    hcl.config.init_dtype = hcl.Float(32)

    l0_is_callable = callable(target)
    l0_is_time_varying = False

    if l0_is_callable:
        print("Detected callable target function. Evaluating dynamically at solver micro-steps.")
        init_target = target(float(tau[0]))
    elif isinstance(target, np.ndarray) and target.ndim > grid.dims:
        l0_is_time_varying = True
        init_target = target[..., 0]
    else:
        init_target = target

    init_target = np.asarray(init_target, dtype=np.float32)
    if constraint is None:
        init_value = init_target
    else:
        constraint = np.asarray(constraint, dtype=np.float32)
        if constraint.ndim > grid.dims:
            constraint_i = constraint[..., 0]
        else:
            constraint_i = constraint
        init_value = np.maximum(init_target, -constraint_i).astype(np.float32)

    max_abs_value = float(compMethod.get("max_abs_value", 1e4))
    value_clip_lo = compMethod.get("value_clip_lo", None)
    value_clip_hi = compMethod.get("value_clip_hi", None)

    # Check only the initial VALUE FUNCTION (not Hamiltonian)
    _check_value("initial target", init_target, max_abs_value)
    _check_value("initial value",  init_value,  max_abs_value)

    V_t         = hcl.asarray(init_value)
    Hamiltonian = hcl.asarray(np.zeros(tuple(grid.pts_each_dim), dtype=np.float32))
    delta_t     = hcl.asarray(np.zeros(1, dtype=np.float32))

    if compMethod["TargetSetMode"] not in ["minVWithVTarget", "maxVWithVTarget"]:
        l0 = hcl.asarray(init_value)
    else:
        l0 = hcl.asarray(init_target)

    solve_pde = _build_graph(dynamics_obj, grid, compMethod["TargetSetMode"], accuracy)
    axes = _build_state_axes(grid)

    if saveAllTimeSteps:
        valfuncs = np.zeros(
            np.insert(tuple(grid.pts_each_dim), grid.dims, len(tau)),
            dtype=np.float32,
        )
        valfuncs[..., -1] = V_t.asnumpy()

    execution_time = 0.0
    t_now = float(tau[0])
    sanitized_now_count = 0
    sanitized_mid_count = 0
    euler_fallback_count = 0
    hold_fallback_count = 0
    converged_flag = False
    V_prev_outer = init_value.copy() if untilConvergent else None

    for i in range(1, len(tau)):
        if l0_is_time_varying and not l0_is_callable:
            l0_step = np.asarray(target[..., i], dtype=np.float32)
            _check_value(f"time-varying target slice {i}", l0_step, max_abs_value)
            l0.copyfrom(l0_step)
        elif not l0_is_callable:
            l0_step = l0.asnumpy()

        while t_now <= float(tau[i]) - 1e-9:
            start = time.time()

            if l0_is_callable:
                l0_now = np.asarray(target(t_now), dtype=np.float32)
                _check_value(f"callable target at t={t_now:.6f}", l0_now, max_abs_value)
                l0.copyfrom(l0_now)

            t = hcl.asarray(np.array([t_now], dtype=np.float32))
            _call_graph(solve_pde, grid, Hamiltonian, V_t, axes, delta_t, t, l0)

            dt = float(delta_t.asnumpy()[0])
            if not np.isfinite(dt) or dt <= 0.0:
                raise FloatingPointError(
                    f"Invalid CFL time step returned by the backend: dt={dt}"
                )
            dt = min(float(tau[i]) - t_now, dt)

            # --- Runge-Kutta midpoint corrector ---
            # ham_now and ham_mid are spatial fluxes (dV/dt) — large values
            # are expected and normal.  Do NOT check their magnitude.
            # The important detail is that the intermediate arithmetic must be
            # carried out in float64 and clipped *before* casting back to float32,
            # otherwise ham_mid * dt can overflow and create NaNs/Infs even when
            # the final clipped value should remain bounded.
            ham_now_raw = Hamiltonian.asnumpy()
            ham_now = _sanitize_flux(ham_now_raw, dt=dt, max_abs_value=max_abs_value)
            if not np.isfinite(np.asarray(ham_now_raw)).all():
                sanitized_now_count += 1
            v_now = np.asarray(V_t.asnumpy(), dtype=np.float64)

            # Check value function only
            _check_value("value slice", v_now, max_abs_value)
            lo_clip, hi_clip = _numeric_clip_bounds(value_clip_lo, value_clip_hi, max_abs_value)

            # Midpoint predictor (bounded in float64 before the backend sees it).
            V_mid_pred = _clip_candidate(v_now + ham_now * dt * 0.5, lo_clip, hi_clip)
            if not np.isfinite(V_mid_pred).all():
                V_mid_pred = _clip_candidate(v_now, lo_clip, hi_clip)

            V_tmp = hcl.asarray(np.asarray(V_mid_pred, dtype=np.float32))
            Hamiltonian_tmp = hcl.asarray(np.zeros(tuple(grid.pts_each_dim), dtype=np.float32))
            delta_t_tmp = hcl.asarray(np.zeros(1, dtype=np.float32))

            t_mid_val = t_now + 0.5 * dt
            t_tmp = hcl.asarray(np.array([t_mid_val], dtype=np.float32))

            if l0_is_callable:
                l0_mid = np.asarray(target(t_mid_val), dtype=np.float32)
                _check_value(f"callable target at t_mid={t_mid_val:.6f}", l0_mid, max_abs_value)
                l0.copyfrom(l0_mid)

            _call_graph(solve_pde, grid, Hamiltonian_tmp, V_tmp, axes, delta_t_tmp, t_tmp, l0)
            ham_mid_raw = Hamiltonian_tmp.asnumpy()
            ham_mid = _sanitize_flux(ham_mid_raw, dt=dt, max_abs_value=max_abs_value)
            if not np.isfinite(np.asarray(ham_mid_raw)).all():
                sanitized_mid_count += 1

            if l0_is_callable:
                l0_step_next = np.asarray(target(t_now + dt), dtype=np.float32)
                _check_value(
                    f"callable target at t={t_now + dt:.6f}", l0_step_next, max_abs_value
                )
            else:
                l0_step_next = l0_step

            gamma_val = float(compMethod.get("cbf_gamma", 0.0))
            rk2_candidate = 0.5 * (v_now + _clip_candidate(v_now + ham_mid * dt, lo_clip, hi_clip))
            rk2_candidate = _finalize_candidate(
                rk2_candidate,
                target_mode=compMethod["TargetSetMode"],
                gamma_val=gamma_val,
                dt=dt,
                l0_step_next=l0_step_next,
                lo=lo_clip,
                hi=hi_clip,
            )

            if np.isfinite(rk2_candidate).all():
                V_next = rk2_candidate
            else:
                # Fallback to the monotone first-order step if the midpoint stage
                # still becomes numerically unstable on a particular micro-step.
                euler_candidate = _finalize_candidate(
                    v_now + ham_now * dt,
                    target_mode=compMethod["TargetSetMode"],
                    gamma_val=gamma_val,
                    dt=dt,
                    l0_step_next=l0_step_next,
                    lo=lo_clip,
                    hi=hi_clip,
                )
                if np.isfinite(euler_candidate).all():
                    euler_fallback_count += 1
                    V_next = euler_candidate
                else:
                    # Last-resort hold step: keep the previous value slice and apply
                    # only the maxCBF clamp / hard bounds. This is conservative but
                    # prevents isolated backend pathologies from killing the solve.
                    hold_fallback_count += 1
                    V_next = _finalize_candidate(
                        v_now,
                        target_mode=compMethod["TargetSetMode"],
                        gamma_val=gamma_val,
                        dt=dt,
                        l0_step_next=l0_step_next,
                        lo=lo_clip,
                        hi=hi_clip,
                    )

            _check_value("updated value slice", V_next, max_abs_value)
            V_t = hcl.asarray(np.asarray(V_next, dtype=np.float32))
            t_now = min(t_now + dt, float(tau[i]))
            execution_time += time.time() - start

        if saveAllTimeSteps:
            valfuncs[..., -1 - i] = V_t.asnumpy()

        # ---- Convergence check (untilConvergent mode) ----
        if untilConvergent:
            V_curr = V_t.asnumpy()
            conv_diff = float(np.max(np.abs(
                np.asarray(V_curr, dtype=np.float64) -
                np.asarray(V_prev_outer, dtype=np.float64)
            )))
            if conv_diff < epsilon:
                print(
                    f"  [untilConvergent] Converged at simulated t={float(tau[i]):.2f}s "
                    f"(step {i}/{len(tau)-1}), max_diff={conv_diff:.2e} < epsilon={epsilon:.2e}"
                )
                converged_flag = True
                break
            V_prev_outer = V_curr.copy()

    print("Total computation time (s): {:.5f}".format(execution_time))
    if sanitized_now_count or sanitized_mid_count or euler_fallback_count or hold_fallback_count:
        print(
            "Solver diagnostics: "
            f"sanitized_now={sanitized_now_count}, "
            f"sanitized_mid={sanitized_mid_count}, "
            f"euler_fallbacks={euler_fallback_count}, "
            f"hold_fallbacks={hold_fallback_count}"
        )

    # When untilConvergent=True, return a single converged slice regardless of
    # saveAllTimeSteps.  The caller stores it as a time-invariant (single-slice)
    # CBVFTable, so ∂_t B = 0 and the QP constraint is always feasible inside
    # the certified safe set.
    if untilConvergent:
        if not converged_flag:
            print(
                f"  WARNING: [untilConvergent] solver did not reach convergence within "
                f"{len(tau)-1} steps (max tau={float(tau[-1]):.1f}s). "
                f"Last max_diff={conv_diff:.2e}. Consider increasing tau or epsilon."
            )
        out = V_t.asnumpy()
        _check_value("converged value tensor", out, max_abs_value)
        return out  # shape (*grid_shape) – single slice

    if saveAllTimeSteps:
        valfuncs[..., 0] = V_t.asnumpy()
        _check_value("saved value tensor (terminal)",  valfuncs[..., -1], max_abs_value)
        _check_value("saved value tensor (earliest)",  valfuncs[...,  0], max_abs_value)
        return valfuncs

    out = V_t.asnumpy()
    _check_value("final value tensor", out, max_abs_value)
    return out