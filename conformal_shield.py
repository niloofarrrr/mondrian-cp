"""
conformal_shield.py  –  Mondrian conformal CBVF shielding (Dubins car).

This revision applies the latest experimental comments directly in code.

Main changes in this version
----------------------------
1. High-resolution offline CBVF defaults.
   The default grid is now 101×101×101 instead of the much coarser debugging
   grids used earlier. Fast mode remains available for quick checks, but it is
   explicitly not intended for paper-quality conclusions.

2. Stronger initial-condition diagnostics.
   The code now prints B(x0,t0) for the exact Lu-style base start x0=[-2,-2,π/4]
   and also reports certification statistics over the actual calibration and
   evaluation starts. This makes it obvious when the rollout is starting outside
   the certified set, in which case the CBVF theorem does not apply.

3. Finite-horizon convergence diagnostics.
   The implementation remains finite-horizon (so the ∂_t B term is still used),
   but it now reports whether the earliest slices of the offline CBVF table have
   approximately converged. This helps diagnose lookback horizons that are too
   short.

4. Finite-horizon CBVF tables by default.
   The offline table now matches the finite-horizon formulation used in the
   paper and stores the full time axis over the rollout horizon. This makes the
   calibrated right-hand side xi_hat enter the online CBVF-QP directly instead
   of relying only on an activation-threshold heuristic.

5. Paper-consistent Mondrian CP tightening.
   Calibration rollouts contribute one maximum mismatch score to every radial
   operating region they visit.  Separate finite-sample conformal quantiles are
   then computed for the effective regions, with a uniform Bonferroni allocation
   of the trajectory-level miscoverage budget. A fixed strict binary region
   tree supplies local parent promotion for sparse branches while data-rich
   branches remain at finer resolution.

6. Region-wise deployment buffers.
   The CP filter selects the maximum calibrated buffer among all effective
   regions intersected by the one-step speed-envelope tube.  The same scalar
   CBVF-QP is retained; only its right-hand side changes from zero to the active
   regional buffer plus the optional inter-sample margin.

7. Paper-demonstration mismatch and baseline defaults.
   The scenario names now match the paper narrative: AlignedModel, HardMismatch,
   and EasyMismatch. HardMismatch is deliberately optimistic in the model
   (default β_model=0.90, β_truth=0.40) so the least-restrictive CBVF baseline
   can fail under true dynamics while the CP filter acts earlier through xi_hat.

8. Multi-seed reporting.
   The default run evaluates several seeds and reports mean ± std across
   seeds for the key metrics. This avoids relying on a single lucky seed.

9. Matched-policy deployment protocol.
   The command-line argument --eval-variants selects which deployment setting
   to evaluate. The default is now the aligned/no-mismatch case only, so an
   aligned RL checkpoint is not accidentally evaluated in hard/easy mismatch
   environments. For future matched experiments, run the script separately with
   --eval-variants hard or --eval-variants easy and the corresponding RL
   checkpoint.
"""

import argparse
import importlib.util
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
if "MPLBACKEND" not in os.environ:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ===========================================================================
# Utilities
# ===========================================================================

def wrap_angle(a: float) -> float:
    """Wrap angle to (-π, π]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def rk4_step(f, x: np.ndarray, dt: float) -> np.ndarray:
    """Fourth-order Runge-Kutta step; wraps the heading angle afterwards."""
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    x_next = x_next.copy()
    x_next[2] = wrap_angle(x_next[2])
    return x_next


# ===========================================================================
# Data classes
# ===========================================================================

@dataclass(frozen=True)
class Interval:
    lo: float
    hi: float

    def clip(self, value: float) -> float:
        return float(np.clip(value, self.lo, self.hi))


@dataclass(frozen=True)
class World:
    xmin: float = -4.0
    xmax: float = 4.0
    ymin: float = -4.0
    ymax: float = 4.0

    goal: Tuple[float, float] = (2.0, 2.0)
    goal_radius: float = 0.30

    obstacle_center: Tuple[float, float] = (0.0, 0.0)
    obstacle_radius: float = 0.50
    robot_radius: float = 0.20

    # Lu-style fixed start for the Dubins benchmark.
    # Small jitter can be added at runtime if desired.
    init_x_range: Tuple[float, float] = (-2.0, -2.0)
    init_y_range: Tuple[float, float] = (-2.0, -2.0)
    init_theta_range: Tuple[float, float] = (math.pi / 4.0, math.pi / 4.0)


@dataclass(frozen=True)
class DynamicsSpec:
    v: float       # maximum forward speed
    beta_u: float  # control-to-turn-rate gain
    v_min: float = 0.0  # braking limit / minimum forward speed


@dataclass(frozen=True)
class DynamicsVariant:
    name: str
    model: DynamicsSpec   # model used for the offline CBVF
    truth: DynamicsSpec   # true plant used in rollouts


@dataclass
class ConstraintTerms:
    B: float
    grad: np.ndarray
    DtB: float
    a_u: np.ndarray        # coefficients of normalized [throttle, yaw]
    base_without_control: float  # DtB + <grad, f_drift> + γ·B


@dataclass
class FilterSolution:
    u: np.ndarray
    feasible: bool
    margin: float          # achieved lhs - originally requested xi
    B: float
    DtB: float
    a_u: np.ndarray
    base_without_control: float
    requested_xi: float
    achieved_lhs: float
    max_achievable_lhs: float
    assumption5_violation: bool



@dataclass(frozen=True)
class ShieldConfig:
    """Activation settings for the baseline CBVF and conformal-CBVF filters.

    The activation threshold controls only when the online filter wakes up.
    The conformal correction xi_hat must enter through the QP right-hand side,
    not through an inflated activation threshold. This matches the calibrated
    CBVF-QP in the paper: CP and CBVF should normally use the same wake-up
    margin, while CP enforces a tighter inequality.
    """

    cbvf_activate_margin: float
    cp_activate_margin: float

    def activation_threshold(self, method: str, xi_hat: float) -> float:
        _ = xi_hat  # xi_hat tightens the QP RHS only; it does not shift wake-up.
        if method == "cp":
            return float(self.cp_activate_margin)
        if method == "cbvf":
            return float(self.cbvf_activate_margin)
        return -float("inf")

    def qp_rhs(self, method: str, xi_hat: float) -> float:
        if method == "cp":
            return float(xi_hat)
        if method == "cbvf":
            return 0.0
        return 0.0

    def describe(self, xi_hat: float) -> Dict[str, float]:
        return {
            "cbvf_activate_margin": float(self.cbvf_activate_margin),
            "cp_activate_margin": float(self.cp_activate_margin),
            "cbvf_activation_threshold": float(self.activation_threshold("cbvf", xi_hat)),
            "cp_activation_threshold": float(self.activation_threshold("cp", xi_hat)),
            "cbvf_qp_rhs": float(self.qp_rhs("cbvf", xi_hat)),
            "cp_qp_rhs": float(self.qp_rhs("cp", xi_hat)),
        }


@dataclass(frozen=True)
class MondrianPartition:
    """Fixed radial partition used by Mondrian conformal calibration.

    The partition variable is the signed clearance from the inflated circular
    obstacle.  If ``clearance_edges=(e_1, ..., e_q)``, the base cells are

        (-inf, e_1), [e_1, e_2), ..., [e_q, +inf).

    This is a measurable state-space partition.  It is fixed before any
    calibration score is inspected and is especially natural for the circular
    Dubins benchmark used here.
    """

    clearance_edges: Tuple[float, ...]
    obstacle_center: Tuple[float, float]
    inflated_obstacle_radius: float

    def __post_init__(self) -> None:
        edges = tuple(float(v) for v in self.clearance_edges)
        if any(not np.isfinite(v) for v in edges):
            raise ValueError("Mondrian clearance edges must be finite.")
        if any(edges[i] >= edges[i + 1] for i in range(len(edges) - 1)):
            raise ValueError("Mondrian clearance edges must be strictly increasing.")
        if self.inflated_obstacle_radius <= 0.0:
            raise ValueError("The inflated obstacle radius must be positive.")
        object.__setattr__(self, "clearance_edges", edges)

    @property
    def n_base_regions(self) -> int:
        return len(self.clearance_edges) + 1

    def clearance(self, state: np.ndarray) -> float:
        px, py = float(state[0]), float(state[1])
        cx, cy = self.obstacle_center
        return float(math.hypot(px - cx, py - cy) - self.inflated_obstacle_radius)

    def base_region(self, state: np.ndarray) -> int:
        return int(np.digitize(self.clearance(state), self.clearance_edges, right=False))

    def candidate_base_regions(
        self,
        state: np.ndarray,
        travel_radius: float,
    ) -> Tuple[int, ...]:
        """Base regions intersected by a conservative one-step position tube.

        Euclidean travel by at most ``travel_radius`` can change obstacle
        clearance by at most the same amount.  We therefore intersect every
        radial cell with that clearance interval.  Boundary-touching cells are
        both retained, which is conservative at region interfaces.
        """
        radius = max(0.0, float(travel_radius))
        center_clearance = self.clearance(state)
        lo = center_clearance - radius
        hi = center_clearance + radius
        bounds = (-float("inf"),) + self.clearance_edges + (float("inf"),)
        candidates: List[int] = []
        for idx in range(self.n_base_regions):
            cell_lo, cell_hi = bounds[idx], bounds[idx + 1]
            # Closed intersection is intentional: exact contact with a region
            # interface retains both adjacent cells for a conservative tube.
            if hi >= cell_lo and lo <= cell_hi:
                candidates.append(idx)
        return tuple(candidates)

    def base_region_label(self, region: int) -> str:
        if not 0 <= int(region) < self.n_base_regions:
            raise IndexError(f"Unknown Mondrian base region {region}.")
        bounds = (-float("inf"),) + self.clearance_edges + (float("inf"),)
        lo, hi = bounds[int(region)], bounds[int(region) + 1]
        if math.isinf(lo):
            return f"clearance < {hi:g}"
        if math.isinf(hi):
            return f"clearance >= {lo:g}"
        return f"{lo:g} <= clearance < {hi:g}"


@dataclass(frozen=True)
class MondrianCalibration:
    """Region-wise buffers defined on a validated cut of a strict region tree."""

    partition: MondrianPartition
    effective_groups: Tuple[Tuple[int, ...], ...]
    base_to_effective: Tuple[int, ...]
    buffers: Tuple[float, ...]
    deltas: Tuple[float, ...]
    counts: Tuple[int, ...]
    delta_traj: float
    epsilon_grid: float
    promotion_rounds: int

    def __post_init__(self) -> None:
        M = len(self.effective_groups)
        if M < 1:
            raise ValueError("Mondrian calibration must have at least one effective region.")
        if not (
            len(self.buffers) == len(self.deltas) == len(self.counts) == M
        ):
            raise ValueError("Mondrian region metadata have inconsistent lengths.")
        if len(self.base_to_effective) != self.partition.n_base_regions:
            raise ValueError("Every base region must map to one effective region.")
        if any(not np.isfinite(v) or v < 0.0 for v in self.buffers):
            raise ValueError("Mondrian buffers must be finite and nonnegative.")
        if any(not 0.0 < float(v) < 1.0 for v in self.deltas):
            raise ValueError("Every effective region needs a valid risk budget.")
        if not 0.0 < float(self.delta_traj) < 1.0:
            raise ValueError("delta_traj must lie strictly between zero and one.")
        if any(int(v) <= 0 for v in self.counts):
            raise ValueError("Every effective region must have calibration scores.")

        expected_bases = set(range(self.partition.n_base_regions))
        covered_bases: List[int] = []
        for group in self.effective_groups:
            if not group:
                raise ValueError("An effective Mondrian region cannot be empty.")
            covered_bases.extend(int(v) for v in group)
        if set(covered_bases) != expected_bases or len(covered_bases) != len(expected_bases):
            raise ValueError(
                "The effective Mondrian regions must be disjoint and cover every base region exactly once."
            )

        expected_map = [-1] * self.partition.n_base_regions
        for effective_region, group in enumerate(self.effective_groups):
            for base_region in group:
                expected_map[int(base_region)] = int(effective_region)
        if tuple(expected_map) != tuple(int(v) for v in self.base_to_effective):
            raise ValueError("base_to_effective is inconsistent with the effective tree cut.")
        # Risk allocators must maintain this exact invariant. The current
        # allocator uses a residual final term and a nextafter correction.
        if math.fsum(float(v) for v in self.deltas) > float(self.delta_traj):
            raise ValueError("Regional risk budgets exceed delta_traj.")

    @property
    def n_effective_regions(self) -> int:
        return len(self.effective_groups)

    @property
    def max_buffer(self) -> float:
        return float(max(self.buffers))

    def effective_region(self, state: np.ndarray) -> int:
        return int(self.base_to_effective[self.partition.base_region(state)])

    def candidate_effective_regions(
        self,
        state: np.ndarray,
        travel_radius: float,
    ) -> Tuple[int, ...]:
        effective = {
            int(self.base_to_effective[base])
            for base in self.partition.candidate_base_regions(state, travel_radius)
        }
        if any(not 0 <= region < self.n_effective_regions for region in effective):
            raise RuntimeError("A reachable base region mapped to an uncalibrated effective region.")
        return tuple(sorted(effective))

    def applied_buffer(
        self,
        state: np.ndarray,
        travel_radius: float,
    ) -> Tuple[float, Tuple[int, ...]]:
        regions = self.candidate_effective_regions(state, travel_radius)
        if not regions:
            raise RuntimeError("The one-step tube did not intersect any Mondrian region.")
        return float(max(self.buffers[m] for m in regions)), regions

    def effective_rollout_scores(
        self,
        raw_base_scores: Dict[int, float],
    ) -> Dict[int, float]:
        """Aggregate raw realized mismatch by the point's own tree-cut region."""
        scores: Dict[int, float] = {}
        for base_region, score in raw_base_scores.items():
            effective_region = int(self.base_to_effective[int(base_region)])
            scores[effective_region] = max(
                scores.get(effective_region, -float("inf")),
                float(score),
            )
        return scores

    def rollout_is_covered(self, raw_base_scores: Dict[int, float]) -> bool:
        """Compare raw realized mismatch with buffers that already contain epsilon_grid."""
        effective_scores = self.effective_rollout_scores(raw_base_scores)
        return all(
            score <= self.buffers[region] + 1e-12
            for region, score in effective_scores.items()
        )

    def as_dict(self) -> Dict[str, object]:
        region_records = []
        for region, group in enumerate(self.effective_groups):
            region_records.append({
                "effective_region": int(region),
                "base_regions": [int(v) for v in group],
                "base_region_labels": [
                    self.partition.base_region_label(v) for v in group
                ],
                "n_scores": int(self.counts[region]),
                "delta_m": float(self.deltas[region]),
                "xi_hat_off": float(self.buffers[region]),
            })
        return {
            "partition": "radial_clearance",
            "clearance_edges": [float(v) for v in self.partition.clearance_edges],
            "n_base_regions": int(self.partition.n_base_regions),
            "n_effective_regions": int(self.n_effective_regions),
            "promotion_rounds": int(self.promotion_rounds),
            "delta_traj": float(self.delta_traj),
            "risk_budget_sum": float(math.fsum(self.deltas)),
            "tree_cut_is_partition": True,
            "tube_region_rule": "closed_interval_including_boundary_touching_neighbors",
            "base_to_effective": [int(v) for v in self.base_to_effective],
            "epsilon_grid": float(self.epsilon_grid),
            "regions": region_records,
        }

# ===========================================================================
# Environment
# ===========================================================================

class DubinsCBVFEnv:
    """
    Dubins-car environment with a circular obstacle.

    State:  x = [px, py, θ]
    Control: u=[throttle,yaw] ∈ [-1,1]^2. Throttle maps affinely to
    bounded forward speed [v_min,v_max], and yaw maps to [-beta,beta].

    This benchmark has no exogenous disturbance variable: D is a singleton and
    the disturbance minimization in the paper is therefore vacuous.
    """

    disturbance_dimension: int = 0

    def __init__(
        self,
        world: World,
        control_bounds: Interval,
        variant: DynamicsVariant,
        dt: float,
        horizon: int,
    ):
        self.world = world
        self.control_bounds = control_bounds
        self.variant = variant
        self.dt = dt
        self.horizon = horizon
        self.state: Optional[np.ndarray] = None
        self.t = 0

    def set_state(self, state: np.ndarray) -> None:
        self.state = state.astype(float).copy()
        self.state[2] = wrap_angle(self.state[2])
        self.t = 0

    def sample_initial_state(
        self, rng: np.random.Generator, cbvf: "CBVFTable", t0: float
    ) -> np.ndarray:
        for _ in range(10_000):
            x = rng.uniform(*self.world.init_x_range)
            y = rng.uniform(*self.world.init_y_range)
            theta = rng.uniform(*self.world.init_theta_range)
            s = np.array([x, y, theta], dtype=float)
            if self.is_safe_state(s):
                B_initial, _, _ = cbvf.value_grad_dt(s, t0)
                if B_initial >= 0.0:
                    return s
        raise RuntimeError(
            "Could not sample a dynamically safe initial state within 10 000 tries."
        )

    def safety_margin_value(self, s: np.ndarray) -> float:
        """Signed-distance safety function l(x).

        Positive values are safe and negative values are unsafe. This matches
        the default CBVF target_shape="signed_distance" used for the offline
        table, so the theoretical safety function and rollout diagnostics are
        consistent.
        """
        x, y = float(s[0]), float(s[1])
        cx, cy = self.world.obstacle_center
        r = self.world.obstacle_radius + self.world.robot_radius
        return float(np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r)

    def is_safe_state(self, s: np.ndarray) -> bool:
        return self.safety_margin_value(s) >= 0.0

    def goal_reached(self, s: np.ndarray) -> bool:
        x, y, _ = s
        gx, gy = self.world.goal
        goal_r = self.world.goal_radius + self.world.robot_radius
        return (x - gx) ** 2 + (y - gy) ** 2 <= goal_r ** 2

    @staticmethod
    def _physical_controls(spec: DynamicsSpec, u: np.ndarray) -> Tuple[float, float]:
        u = np.clip(np.asarray(u, dtype=float).reshape(2), -1.0, 1.0)
        speed = spec.v_min + 0.5 * (u[0] + 1.0) * (spec.v - spec.v_min)
        return float(speed), float(spec.beta_u * u[1])

    def model_dynamics(self, s: np.ndarray, u: np.ndarray) -> np.ndarray:
        """f_app(x, u) – the approximate model used for offline CBVF."""
        _, _, th = s
        spec = self.variant.model
        speed, yaw_rate = self._physical_controls(spec, u)
        return np.array([speed * np.cos(th), speed * np.sin(th), yaw_rate], dtype=float)

    def true_dynamics(self, s: np.ndarray, u: np.ndarray) -> np.ndarray:
        """f(x, u) – the true plant dynamics used in simulation."""
        _, _, th = s
        spec = self.variant.truth
        speed, yaw_rate = self._physical_controls(spec, u)
        return np.array([speed * np.cos(th), speed * np.sin(th), yaw_rate], dtype=float)

    def step(
        self,
        u: np.ndarray,
        terminate_on_unsafe: bool = True,
    ) -> Tuple[np.ndarray, bool, Dict[str, float]]:
        if self.state is None:
            raise RuntimeError("State has not been initialised.")
        u = np.clip(np.asarray(u, dtype=float).reshape(2), self.control_bounds.lo, self.control_bounds.hi)
        self.state = rk4_step(lambda z: self.true_dynamics(z, u), self.state, self.dt)
        self.t += 1
        unsafe = not self.is_safe_state(self.state)
        goal = self.goal_reached(self.state)
        done = goal or (self.t >= self.horizon) or (terminate_on_unsafe and unsafe)
        return self.state.copy(), done, {"unsafe": float(unsafe), "goal": float(goal)}


# ===========================================================================
# Nominal policy interface
# ===========================================================================

try:
    from train.rl_nominal_policy import RLNominalPolicy, HeuristicNominalPolicy
except Exception as exc:  # delayed error if the RL files are missing
    RLNominalPolicy = None
    HeuristicNominalPolicy = None
    _RL_IMPORT_ERROR = exc
else:
    _RL_IMPORT_ERROR = None


def _parse_hidden_dims(text: str) -> Tuple[int, ...]:
    values = tuple(int(v.strip()) for v in str(text).split(",") if v.strip())
    if not values:
        raise ValueError("--rl-hidden-dims must contain at least one integer, e.g. 256,256")
    return values


def build_nominal_policy(args, world: World, control_bounds: Interval):
    if args.nominal_controller == "rl":
        if RLNominalPolicy is None:
            raise RuntimeError(
                "Could not import train.rl_nominal_policy. Keep the train/ folder "
                "next to conformal_shield.py. Original import error: " + repr(_RL_IMPORT_ERROR)
            )
        return RLNominalPolicy(
            world=world,
            control_bounds=control_bounds,
            checkpoint_dir=args.rl_checkpoint_dir,
            checkpoint_step=args.rl_checkpoint_step,
            seed=args.rl_seed,
            hidden_dims=_parse_hidden_dims(args.rl_hidden_dims),
            deterministic=not args.rl_stochastic_action,
        )

    if HeuristicNominalPolicy is None:
        raise RuntimeError(
            "Could not import the fallback heuristic policy from train.rl_nominal_policy. "
            "Original import error: " + repr(_RL_IMPORT_ERROR)
        )
    return HeuristicNominalPolicy(
        world=world,
        control_bounds=control_bounds,
        k_heading=args.k_heading,
    )


def goal_distance(world: World, s: np.ndarray) -> float:
    gx, gy = world.goal
    return float(np.linalg.norm(np.asarray(s[:2], dtype=float) - np.array([gx, gy], dtype=float)))


# ===========================================================================
# CBVF table (pre-computed offline)
# ===========================================================================

class CBVFTable:
    """
    Stores the offline CBVF  B̃_γ^{f̃}(x, t)  on a 3-D grid  (x, y, θ)  over
    a time axis  tau  with tau[0] = t0 < 0 (earliest time) and tau[-1] = 0
    (terminal time).

    valfuncs[..., k] is the value at physical time tau[k], so k=0 is the
    deepest backward slice and k=-1 is the terminal condition l(x).

    The temporal derivative  ∂_t B  is computed as a forward finite difference
    toward the terminal time (increasing k), consistent with the CBVF
    variational inequality.
    """

    def __init__(
        self,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        th_grid: np.ndarray,
        tau: np.ndarray,
        B_table: np.ndarray,
        max_abs_table: float = 10.0,
        max_abs_grad: float = 100.0,
    ):
        self.x_grid = np.asarray(x_grid, dtype=float)
        self.y_grid = np.asarray(y_grid, dtype=float)
        self.th_grid = np.asarray(th_grid, dtype=float)
        self.tau = np.asarray(tau, dtype=float)      # tau[0]=t0<0, tau[-1]=0
        self.B_table = np.asarray(B_table, dtype=float)

        expected_shape = (
            len(self.x_grid),
            len(self.y_grid),
            len(self.th_grid),
            len(self.tau),
        )
        if self.B_table.shape != expected_shape:
            raise ValueError(
                f"B_table shape {self.B_table.shape} does not match expected {expected_shape}."
            )
        # len(tau)==1 is valid: it marks a converged time-invariant table.
        # len(tau)>=2 is required only for finite-horizon multi-slice tables.
        if len(self.tau) < 1:
            raise ValueError("tau must contain at least one time point.")
        if not np.isfinite(self.B_table).all():
            raise ValueError("CBVF table contains non-finite values.")
        if float(np.max(np.abs(self.B_table))) > max_abs_table:
            raise ValueError(
                f"CBVF table magnitude is implausibly large "
                f"(max abs {float(np.max(np.abs(self.B_table))):.3e})."
            )

        # For single-slice (time-invariant) tables dt is unused; set to 1.0
        # as a safe sentinel so no code divides by zero.
        self.dt = float(self.tau[1] - self.tau[0]) if len(self.tau) >= 2 else 1.0

        # Pre-compute spatial gradients at each time slice
        self.gx = np.empty_like(self.B_table)
        self.gy = np.empty_like(self.B_table)
        self.gth = np.empty_like(self.B_table)

        for k in range(len(self.tau)):
            gx_k, gy_k, gth_k = np.gradient(
                self.B_table[..., k],
                self.x_grid,
                self.y_grid,
                self.th_grid,
                edge_order=2,
            )
            if not (
                np.isfinite(gx_k).all()
                and np.isfinite(gy_k).all()
                and np.isfinite(gth_k).all()
            ):
                raise ValueError(f"Non-finite spatial gradients on time slice {k}.")
            slice_grad_max = max(
                float(np.max(np.abs(gx_k))),
                float(np.max(np.abs(gy_k))),
                float(np.max(np.abs(gth_k))),
            )
            if slice_grad_max > max_abs_grad:
                raise ValueError(
                    f"CBVF spatial gradients are implausibly large on slice {k} "
                    f"(max abs {slice_grad_max:.3e})."
                )
            self.gx[..., k] = gx_k
            self.gy[..., k] = gy_k
            self.gth[..., k] = gth_k

    def convergence_stats(self, n_tail_slices: int = 5) -> Dict[str, float]:
        """Diagnostics for whether the finite-horizon table has approximately converged at t0."""
        if len(self.tau) == 1:
            # Single-slice table produced by untilConvergent solver: already converged.
            return {
                "earliest_slice_abs_diff": 0.0,
                "earliest_slice_rel_diff": 0.0,
                "tail_slice_abs_diff_mean": 0.0,
                "tail_slice_abs_diff_max": 0.0,
                "n_tail_pairs": 0.0,
                "time_invariant": 1.0,
            }
        n_pairs = max(1, min(int(n_tail_slices), len(self.tau) - 1))
        diffs = []
        for k in range(n_pairs):
            diffs.append(float(np.max(np.abs(self.B_table[..., k] - self.B_table[..., k + 1]))))
        base_scale = max(float(np.max(np.abs(self.B_table[..., 0]))), 1e-8)
        earliest_abs = float(diffs[0])
        return {
            "earliest_slice_abs_diff": earliest_abs,
            "earliest_slice_rel_diff": float(earliest_abs / base_scale),
            "tail_slice_abs_diff_mean": float(np.mean(diffs)),
            "tail_slice_abs_diff_max": float(np.max(diffs)),
            "n_tail_pairs": float(n_pairs),
            "time_invariant": 0.0,
        }

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    @staticmethod
    def from_npz(
        path: str,
        max_abs_table: float = 10.0,
        max_abs_grad: float = 100.0,
    ) -> "CBVFTable":
        data = np.load(path)
        return CBVFTable(
            x_grid=data["x_grid"],
            y_grid=data["y_grid"],
            th_grid=data["th_grid"],
            tau=data["tau"],
            B_table=data["B_table"],
            max_abs_table=max_abs_table,
            max_abs_grad=max_abs_grad,
        )

    # ------------------------------------------------------------------
    # Interpolation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cell_and_weight(grid: np.ndarray, value: float) -> Tuple[int, float]:
        value = float(np.clip(value, grid[0], grid[-1]))
        j = int(np.searchsorted(grid, value, side="right") - 1)
        j = max(0, min(j, len(grid) - 2))
        g0, g1 = grid[j], grid[j + 1]
        w = 0.0 if g1 == g0 else (value - g0) / (g1 - g0)
        return j, w

    def _nearest_time_index(self, t: float) -> int:
        return int(np.argmin(np.abs(self.tau - t)))

    def _interp3(self, field3: np.ndarray, s: np.ndarray) -> float:
        x, y, th = float(s[0]), float(s[1]), wrap_angle(float(s[2]))
        ix, wx = self._cell_and_weight(self.x_grid, x)
        iy, wy = self._cell_and_weight(self.y_grid, y)
        it, wt = self._cell_and_weight(self.th_grid, th)

        c000 = field3[ix,     iy,     it    ]
        c001 = field3[ix,     iy,     it + 1]
        c010 = field3[ix,     iy + 1, it    ]
        c011 = field3[ix,     iy + 1, it + 1]
        c100 = field3[ix + 1, iy,     it    ]
        c101 = field3[ix + 1, iy,     it + 1]
        c110 = field3[ix + 1, iy + 1, it    ]
        c111 = field3[ix + 1, iy + 1, it + 1]

        c00 = (1.0 - wx) * c000 + wx * c100
        c01 = (1.0 - wx) * c001 + wx * c101
        c10 = (1.0 - wx) * c010 + wx * c110
        c11 = (1.0 - wx) * c011 + wx * c111
        c0 = (1.0 - wy) * c00 + wy * c10
        c1 = (1.0 - wy) * c01 + wy * c11
        return float((1.0 - wt) * c0 + wt * c1)

    def value_grad_dt(
        self, s: np.ndarray, t: float
    ) -> Tuple[float, np.ndarray, float]:
        """
        Return  (B(x,t),  ∇_x B(x,t),  ∂_t B(x,t)).

        For time-invariant (converged) tables (len(tau)==1):
          - t is ignored
          - DtB = 0  (steady-state condition of the converged CBVF)
          - The QP constraint reduces to <∇B, f> + γ·B ≥ ξ

        For multi-slice finite-horizon tables:
          - DtB uses a forward difference toward the terminal time.
        """
        if len(self.tau) == 1:
            # Time-invariant converged CBVF: single slice, DtB = 0
            B = self._interp3(self.B_table[..., 0], s)
            grad = np.array(
                [
                    self._interp3(self.gx[..., 0], s),
                    self._interp3(self.gy[..., 0], s),
                    self._interp3(self.gth[..., 0], s),
                ],
                dtype=float,
            )
            return B, grad, 0.0

        k = self._nearest_time_index(t)
        B = self._interp3(self.B_table[..., k], s)
        grad = np.array(
            [
                self._interp3(self.gx[..., k], s),
                self._interp3(self.gy[..., k], s),
                self._interp3(self.gth[..., k], s),
            ],
            dtype=float,
        )

        # Forward difference: ∂_t B ≈ (B(t + Δt) - B(t)) / Δt
        # (toward terminal time = increasing k)
        if k < len(self.tau) - 1:
            B_fwd = self._interp3(self.B_table[..., k + 1], s)
            DtB = (B_fwd - B) / self.dt
        else:
            # At the terminal slice use backward difference
            B_bwd = self._interp3(self.B_table[..., k - 1], s)
            DtB = (B - B_bwd) / self.dt

        return B, grad, float(DtB)


# ===========================================================================
# ODP dynamics class (used by the HJ solver during offline computation)
# ===========================================================================

try:
    import heterocl as hcl
except ImportError:  # pragma: no cover
    hcl = None


class ModelDubinsCBVFODP:
    """
    Dubins-car dynamics expressed in the HeteroCL interface required by
    optimized_dp.  The controller maximises the CBVF (uMode="max") and there
    is no explicit disturbance.
    """

    def __init__(
        self,
        speed: float,
        speed_min: float,
        beta_u: float,
        control_bounds: Interval,
        uMode: str = "max",
        dMode: str = "min",
    ):
        self.speed = float(speed)
        self.speed_min = float(speed_min)
        self.beta_u = float(beta_u)
        self.u_lo = float(control_bounds.lo)
        self.u_hi = float(control_bounds.hi)
        self.uMode = uMode
        self.dMode = dMode

    def opt_ctrl(self, t, state, spat_deriv):
        opt_throttle = hcl.scalar(self.u_hi, "opt_throttle")
        opt_yaw = hcl.scalar(self.u_hi, "opt_yaw")
        in4 = hcl.scalar(0.0, "in4")

        speed_half = 0.5 * (self.speed - self.speed_min)
        coeff_throttle = speed_half * (
            hcl.cos(state[2]) * spat_deriv[0]
            + hcl.sin(state[2]) * spat_deriv[1]
        )
        coeff_yaw = self.beta_u * spat_deriv[2]
        if self.uMode == "max":
            with hcl.if_(coeff_throttle >= 0):
                opt_throttle[0] = self.u_hi
            with hcl.else_():
                opt_throttle[0] = self.u_lo
            with hcl.if_(coeff_yaw >= 0):
                opt_yaw[0] = self.u_hi
            with hcl.else_():
                opt_yaw[0] = self.u_lo
        elif self.uMode == "min":
            with hcl.if_(coeff_throttle >= 0):
                opt_throttle[0] = self.u_lo
            with hcl.else_():
                opt_throttle[0] = self.u_hi
            with hcl.if_(coeff_yaw >= 0):
                opt_yaw[0] = self.u_lo
            with hcl.else_():
                opt_yaw[0] = self.u_hi
        else:
            raise ValueError(f"Unsupported uMode {self.uMode}")

        return (opt_throttle[0], opt_yaw[0], in4[0])

    def opt_dstb(self, t, state, spat_deriv):
        d1 = hcl.scalar(0.0, "d1")
        d2 = hcl.scalar(0.0, "d2")
        d3 = hcl.scalar(0.0, "d3")
        return (d1[0], d2[0], d3[0])

    def dynamics(self, t, state, uOpt, dOpt):
        x_dot = hcl.scalar(0.0, "x_dot")
        y_dot = hcl.scalar(0.0, "y_dot")
        theta_dot = hcl.scalar(0.0, "theta_dot")
        physical_speed = self.speed_min + 0.5 * (uOpt[0] + 1.0) * (self.speed - self.speed_min)
        x_dot[0] = physical_speed * hcl.cos(state[2])
        y_dot[0] = physical_speed * hcl.sin(state[2])
        theta_dot[0] = self.beta_u * uOpt[1]
        return (x_dot[0], y_dot[0], theta_dot[0])


# ===========================================================================
# Target / obstacle function builders
# ===========================================================================

def build_world_target_on_odp_grid(
    world: World,
    grid,
    target_shape: str,
    target_clip: float,
    target_scale: float,
) -> np.ndarray:
    """
    Build the bounded obstacle target  l(x)  on the ODP grid.
    The zero level set  {l = 0}  matches the obstacle boundary.
    """
    X = np.asarray(grid.vs[0], dtype=np.float32)
    Y = np.asarray(grid.vs[1], dtype=np.float32)
    cx, cy = world.obstacle_center
    r = float(world.obstacle_radius + world.robot_radius)

    radial_sq = (X - cx) ** 2 + (Y - cy) ** 2
    signed_sq = radial_sq - r ** 2

    if target_shape == "squared":
        bounded = np.clip(signed_sq, -float(target_clip), float(target_clip))
    elif target_shape == "signed_distance":
        bounded = np.clip(np.sqrt(radial_sq) - r, -float(target_clip), float(target_clip))
    else:
        raise ValueError(f"Unsupported target_shape: {target_shape}")

    return np.asarray(bounded, dtype=np.float32)


# ===========================================================================
# Module / file loading helpers
# ===========================================================================

def _load_grid_class(odp_root: Optional[str], script_dir: Path):
    import_candidates = []
    if odp_root is not None:
        import_candidates.append(Path(odp_root).expanduser().resolve())

    import_candidates.extend(
        [
            script_dir,
            script_dir / "optimized_dp",
            script_dir.parent,
            script_dir.parent / "optimized_dp",
            Path.home() / "Desktop" / "optimized_dp",
            Path.home() / "optimized_dp",
        ]
    )

    seen: set = set()
    for candidate in import_candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "odp").exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))

    try:
        from odp.Grid.GridProcessing import Grid
        return Grid
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import 'odp'. Pass --odp-root to the folder containing "
            "the 'odp' package."
        ) from exc


def _load_local_hjsolver(solver_file: Optional[str], script_dir: Path):
    if solver_file is not None:
        solver_path = Path(solver_file).expanduser()
    else:
        default_candidates = [
            script_dir / "solver_cbvf.py",
            script_dir / "solver_cbvf(20).py",
        ]
        solver_path = next(
            (candidate for candidate in default_candidates if candidate.exists()),
            default_candidates[0],
        )
    if not solver_path.is_absolute():
        solver_path = (script_dir / solver_path).resolve()
    else:
        solver_path = solver_path.resolve()

    if not solver_path.exists():
        raise FileNotFoundError(
            f"Could not find solver file at {solver_path}. "
            "Pass --solver-file with the exact path."
        )
    print(f"Using local HJ solver file: {solver_path}")

    spec = importlib.util.spec_from_file_location(
        "local_solver_cbvf_module", str(solver_path)
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load solver module from {solver_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "HJSolver"):
        raise AttributeError(
            f"Solver file {solver_path} does not define HJSolver"
        )
    return module.HJSolver


# ===========================================================================
# CBVF table computation and caching
# ===========================================================================

def _table_cache_key(
    model_spec: DynamicsSpec,
    gamma: float,
    cbvf_dt: float,
    nx: int,
    ny: int,
    nth: int,
    target_shape: str,
    target_clip: float,
    target_scale: float,
    lookback_length: float = 0.0,
) -> str:
    # lookback_length=0.0 marks a converged time-invariant table.
    return (
        f"vmin{model_spec.v_min:.3f}_vmax{model_spec.v:.3f}_bu{model_spec.beta_u:.3f}_"
        f"g{gamma:.3f}_cbvfdt{cbvf_dt:.3f}_lb{lookback_length:.1f}_"
        f"nx{nx}_ny{ny}_nth{nth}_"
        f"target{target_shape}_clip{target_clip:.2f}_scale{target_scale:.2f}"
    )


def compute_actual_cbvf_npz(
    output_path: Path,
    world: World,
    control_bounds: Interval,
    model_spec: DynamicsSpec,
    gamma: float,
    cbvf_dt: float,
    nx: int,
    ny: int,
    nth: int,
    accuracy: str,
    odp_root: Optional[str],
    solver_file: Optional[str],
    target_shape: str,
    target_clip: float,
    target_scale: float,
    max_abs_solver_value: float,
    lookback_length: float,
    use_time_invariant_cbvf: bool,
) -> Path:
    script_dir = Path(__file__).resolve().parent
    Grid = _load_grid_class(odp_root=odp_root, script_dir=script_dir)
    HJSolver = _load_local_hjsolver(solver_file=solver_file, script_dir=script_dir)

    grid = Grid(
        minBounds=np.array([world.xmin, world.ymin, -math.pi], dtype=float),
        maxBounds=np.array([world.xmax, world.ymax, math.pi], dtype=float),
        dims=3,
        pts_each_dim=np.array([nx, ny, nth], dtype=int),
        periodicDims=[2],
    )

    dynamics = ModelDubinsCBVFODP(
        speed=model_spec.v,
        speed_min=model_spec.v_min,
        beta_u=model_spec.beta_u,
        control_bounds=control_bounds,
        uMode="max",
        dMode="min",
    )

    solver_attempts = [
        {
            "label": "requested settings",
            "gamma": float(gamma),
            "cbvf_dt": float(cbvf_dt),
            "target_shape": str(target_shape),
            "target_clip": float(target_clip),
            "target_scale": float(target_scale),
            "accuracy": str(accuracy),
        },
        {
            "label": "signed-distance fallback",
            "gamma": min(float(gamma), 0.10),
            "cbvf_dt": float(cbvf_dt),
            "target_shape": "signed_distance",
            "target_clip": float(target_clip),
            "target_scale": float(target_scale),
            "accuracy": "low",
        },
        {
            "label": "signed-distance + larger dt",
            "gamma": min(float(gamma), 0.05),
            "cbvf_dt": max(float(cbvf_dt), 0.25),
            "target_shape": "signed_distance",
            "target_clip": float(target_clip),
            "target_scale": float(target_scale),
            "accuracy": "low",
        },
    ]

    CONVERGENCE_MAX_SIM_TIME = 600.0
    CONVERGENCE_EPSILON = 1e-3

    B_table = None
    last_exc = None
    tau_solver = None

    for attempt_idx, attempt in enumerate(solver_attempts, start=1):
        if attempt_idx > 1:
            print(
                f"  retrying offline CBVF solve ({attempt_idx}/{len(solver_attempts)}): "
                f"{attempt['label']} "
                f"[target={attempt['target_shape']}, gamma={attempt['gamma']}, "
                f"cbvf_dt={attempt['cbvf_dt']}, accuracy={attempt['accuracy']}]"
            )
        try:
            target_values = build_world_target_on_odp_grid(
                world=world,
                grid=grid,
                target_shape=attempt["target_shape"],
                target_clip=attempt["target_clip"],
                target_scale=attempt["target_scale"],
            )
            if use_time_invariant_cbvf:
                tau_solver = np.arange(
                    0.0, CONVERGENCE_MAX_SIM_TIME + 1e-9, attempt["cbvf_dt"], dtype=float
                )
            else:
                tau_solver = np.arange(0.0, lookback_length + 1e-9, attempt["cbvf_dt"], dtype=float)
                if len(tau_solver) < 2:
                    tau_solver = np.array([0.0, max(float(lookback_length), float(attempt["cbvf_dt"]))], dtype=float)

            compMethod = {
                "TargetSetMode": "maxCBF",
                "cbf_gamma": float(attempt["gamma"]),
                "value_clip_lo": -float(attempt["target_clip"]),
                "value_clip_hi": float(attempt["target_clip"]),
                "max_abs_value": float(max_abs_solver_value),
            }
            B_raw = HJSolver(
                dynamics_obj=dynamics,
                grid=grid,
                multiple_value=target_values,
                tau=tau_solver,
                compMethod=compMethod,
                saveAllTimeSteps=not use_time_invariant_cbvf,
                accuracy=attempt["accuracy"],
                untilConvergent=use_time_invariant_cbvf,
                epsilon=CONVERGENCE_EPSILON,
            )
            B_table = B_raw
            break
        except FloatingPointError as exc:
            last_exc = exc
            B_table = None

    if B_table is None or tau_solver is None:
        raise FloatingPointError(
            "Offline CBVF solve failed under all stabilisation attempts. "
            f"Last error: {last_exc}."
        )
    if not np.isfinite(B_table).all():
        raise FloatingPointError(
            "Offline CBVF solver returned a non-finite value table."
        )

    if use_time_invariant_cbvf:
        tau_physical = np.array([0.0], dtype=np.float32)
        B_table_aligned = np.asarray(B_table, dtype=np.float32)[..., np.newaxis]
    else:
        lookback = float(tau_solver[-1])
        tau_physical = np.linspace(-lookback, 0.0, len(tau_solver), dtype=np.float32)
        B_table_aligned = np.asarray(B_table, dtype=np.float32)

    np.savez(
        output_path,
        x_grid=np.asarray(grid.grid_points[0], dtype=np.float32),
        y_grid=np.asarray(grid.grid_points[1], dtype=np.float32),
        th_grid=np.asarray(grid.grid_points[2], dtype=np.float32),
        tau=tau_physical,
        B_table=B_table_aligned,
    )
    return output_path


def try_load_table(
    cbvf_path: Path,
    max_abs_table: float,
    max_abs_grad: float,
) -> Optional[CBVFTable]:
    try:
        return CBVFTable.from_npz(
            str(cbvf_path),
            max_abs_table=max_abs_table,
            max_abs_grad=max_abs_grad,
        )
    except Exception as exc:
        print(f"Cached CBVF table at {cbvf_path} failed sanity checks: {exc}")
        return None



def prepare_cbvf_table(
    cache_dir: Path,
    world: World,
    control_bounds: Interval,
    model_spec: DynamicsSpec,
    gamma: float,
    dt: float,
    horizon: int,
    cbvf_dt: float,
    nx: int,
    ny: int,
    nth: int,
    accuracy: str,
    recompute: bool,
    odp_root: Optional[str],
    solver_file: Optional[str],
    target_shape: str,
    target_clip: float,
    target_scale: float,
    max_abs_table: float,
    max_abs_grad: float,
    max_abs_solver_value: float,
    allow_coarse_fallback: bool,
    max_est_runtime_gb: float,
    use_time_invariant_cbvf: bool,
) -> Tuple[CBVFTable, bool, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)

    grid_attempts = [(int(nx), int(ny), int(nth))]
    if allow_coarse_fallback:
        for cand in [
            (min(int(nx), 81), min(int(ny), 81), min(int(nth), 61)),
            (min(int(nx), 71), min(int(ny), 71), min(int(nth), 51)),
            (min(int(nx), 61), min(int(ny), 61), min(int(nth), 41)),
            (min(int(nx), 51), min(int(ny), 51), min(int(nth), 41)),
        ]:
            if cand not in grid_attempts:
                grid_attempts.append(cand)

    cache_lookback = 0.0 if use_time_invariant_cbvf else float(horizon) * float(dt)
    last_error = None

    for grid_idx, (nx_i, ny_i, nth_i) in enumerate(grid_attempts, start=1):
        name = (
            f"cbvf_table_"
            f"{_table_cache_key(model_spec, gamma, cbvf_dt, nx_i, ny_i, nth_i, target_shape, target_clip, target_scale, cache_lookback)}.npz"
        )
        cbvf_path = cache_dir / name

        if not recompute and cbvf_path.exists():
            loaded = try_load_table(cbvf_path, max_abs_table=max_abs_table, max_abs_grad=max_abs_grad)
            if loaded is not None:
                mode_label = "converged" if use_time_invariant_cbvf else "finite-horizon"
                print(f"Loading precomputed {mode_label} CBVF table from {cbvf_path}")
                return loaded, False, cbvf_path
            print("Recomputing the cached table because it is numerically corrupted.")

        n_slices = 2 if use_time_invariant_cbvf else int(math.ceil(cache_lookback / float(cbvf_dt))) + 1
        values_only_gib = int(nx_i) * int(ny_i) * int(nth_i) * int(n_slices) * 4 / (1024 ** 3)
        est_runtime_gib = values_only_gib * (5.0 if not use_time_invariant_cbvf else 1.0)

        if not use_time_invariant_cbvf and est_runtime_gib > float(max_est_runtime_gb):
            last_error = RuntimeError(
                "Requested offline CBVF table is likely too large for this machine "
                f"(values-only ~{values_only_gib:.2f} GiB, estimated runtime footprint ~{est_runtime_gib:.2f} GiB)."
            )
            print(
                f"Skipping requested grid {nx_i}x{ny_i}x{nth_i}: {last_error} "
                "Use --allow-coarse-fallback, reduce the grid, increase --cbvf-dt, "
                "shorten --horizon, or use --time-invariant-cbvf."
            )
            continue

        if grid_idx == 1:
            mode_label = "converged (time-invariant)" if use_time_invariant_cbvf else "finite-horizon"
            print(f"Building {mode_label} CBVF table at {cbvf_path} ...")
        else:
            print(f"Retrying with coarser grid {nx_i}x{ny_i}x{nth_i} at {cbvf_path} ...")

        if use_time_invariant_cbvf:
            est_mb = int(nx_i) * int(ny_i) * int(nth_i) * 2 * 4 / (1024 ** 2)
            print(
                f"  grid: {nx_i}×{ny_i}×{nth_i}  "
                f"live RAM: 2 slices ≈ {est_mb:.1f} MiB  "
                f"(solver runs until max|ΔV| < 1e-3, up to 600 s simulated time)"
            )
        else:
            print(
                f"  offline table size estimate: {nx_i}x{ny_i}x{nth_i} over {n_slices} times "
                f"(~{values_only_gib:.2f} GiB for values only, estimated runtime footprint ~{est_runtime_gib:.2f} GiB)"
            )

        try:
            compute_actual_cbvf_npz(
                output_path=cbvf_path,
                world=world,
                control_bounds=control_bounds,
                model_spec=model_spec,
                gamma=gamma,
                cbvf_dt=cbvf_dt,
                nx=nx_i,
                ny=ny_i,
                nth=nth_i,
                accuracy=accuracy,
                odp_root=odp_root,
                solver_file=solver_file,
                target_shape=target_shape,
                target_clip=target_clip,
                target_scale=target_scale,
                max_abs_solver_value=max_abs_solver_value,
                lookback_length=cache_lookback,
                use_time_invariant_cbvf=use_time_invariant_cbvf,
            )
            loaded = try_load_table(cbvf_path, max_abs_table=max_abs_table, max_abs_grad=max_abs_grad)
            if loaded is not None:
                return loaded, True, cbvf_path
            last_error = RuntimeError(
                "A freshly generated CBVF table still failed sanity checks. "
                "Trying a coarser grid."
            )
        except Exception as exc:
            last_error = exc
            print(f"  grid attempt {nx_i}x{ny_i}x{nth_i} failed: {exc}")

    raise RuntimeError(
        "Offline CBVF table generation failed on all grid attempts. "
        f"Last error: {last_error}"
    )


# ===========================================================================
# CBVF constraint evaluation
# ===========================================================================

def _require_disturbance_free_benchmark(
    env: DubinsCBVFEnv,
    caller: str,
) -> None:
    """Fail loudly if disturbance-aware dynamics are introduced later.

    The closed-form constraint and mismatch formulas below omit the paper's
    minimization over d because this Dubins benchmark declares D to be a
    singleton. They must not be reused silently for a disturbed environment.
    """
    if getattr(env, "disturbance_dimension", None) != 0:
        raise NotImplementedError(
            f"{caller} is implemented only for the disturbance-free Dubins "
            "benchmark. Add the explicit min over the disturbance set before "
            "using a disturbed dynamics model."
        )

def model_inner_product(
    s: np.ndarray, u: np.ndarray, grad: np.ndarray, env: DubinsCBVFEnv
) -> float:
    """<∇B, f_app(x, u)> — model directional derivative."""
    _require_disturbance_free_benchmark(env, "model_inner_product")
    return float(np.dot(grad, env.model_dynamics(s, u)))


def model_constraint_terms(
    s: np.ndarray,
    t: float,
    cbvf: CBVFTable,
    env: DubinsCBVFEnv,
    gamma: float,
) -> ConstraintTerms:
    """
    Evaluate all terms of the CBVF constraint at (s, t):

        ∂_t B + <∇B, f_app(x, u, d*)> + γ·B ≥ ξ̂

    For a control-affine system with no disturbance the constraint is affine
    in u.  We split it as:

        base_without_control + a_u · u ≥ ξ̂

    where
        base_without_control = ∂_t B + <∇B, f_drift(x)> + γ·B
        a_u[0] = (∇_{xy}B · heading)(v_max-v_min)/2
        a_u[1] = (∂B/∂θ) β_model.
    """
    _require_disturbance_free_benchmark(env, "model_constraint_terms")
    B, grad, DtB = cbvf.value_grad_dt(s, t)
    _, _, th = s
    spec = env.variant.model
    heading_derivative = float(grad[0] * np.cos(th) + grad[1] * np.sin(th))
    speed_center = 0.5 * (spec.v + spec.v_min)
    speed_half = 0.5 * (spec.v - spec.v_min)
    drift_only = float(heading_derivative * speed_center)
    a_u = np.array(
        [heading_derivative * speed_half, grad[2] * spec.beta_u], dtype=float
    )
    base_without_control = float(DtB + drift_only + gamma * B)
    return ConstraintTerms(
        B=float(B),
        grad=grad,
        DtB=float(DtB),
        a_u=a_u,
        base_without_control=base_without_control,
    )


# ===========================================================================
# CBVF-QP projection (Eq. 8 / Eq. 16 in the paper)
# ===========================================================================

def solve_cbvf_projection_exact(
    s: np.ndarray,
    t: float,
    u_nom: np.ndarray,
    cbvf: CBVFTable,
    env: DubinsCBVFEnv,
    gamma: float,
    xi: float,
    tol: float = 1e-10,
) -> FilterSolution:
    """
    Solve the CBVF-QP:

        min_{u ∈ U}  ‖u - u_nom‖²
        s.t.         base_without_control + a_u · u ≥ xi

    Exact Euclidean projection onto a box intersected with one affine
    half-space. The scalar KKT multiplier is found monotonically by bisection.
    If the inequality is infeasible under the actuator bounds, return the
    least-violation control induced by the same affine constraint rather than
    a separate geometric steering heuristic. Feasibility and margin are always
    evaluated against the original requested xi; the target is never silently
    clipped. Such events are surfaced as Assumption-5 violations.
    """
    u_nom = np.clip(np.asarray(u_nom, dtype=float).reshape(2), env.control_bounds.lo, env.control_bounds.hi)
    terms = model_constraint_terms(s=s, t=t, cbvf=cbvf, env=env, gamma=gamma)
    a_u = terms.a_u
    base = terms.base_without_control

    maximizing_u = np.where(a_u >= 0.0, env.control_bounds.hi, env.control_bounds.lo)
    max_control_effort = float(np.dot(a_u, maximizing_u))
    max_achievable_lhs = base + max_control_effort
    requested_xi = float(xi)
    assumption5_violation = bool(max_achievable_lhs < requested_xi - tol)

    if float(np.linalg.norm(a_u)) < tol:
        feasible = bool(base >= requested_xi - tol)
        u = u_nom
        achieved_lhs = float(base + np.dot(a_u, u))
        margin = float(achieved_lhs - requested_xi)
        return FilterSolution(
            u=np.asarray(u, dtype=float),
            feasible=bool(feasible),
            margin=float(margin),
            B=terms.B,
            DtB=terms.DtB,
            a_u=terms.a_u,
            base_without_control=terms.base_without_control,
            requested_xi=requested_xi,
            achieved_lhs=achieved_lhs,
            max_achievable_lhs=float(max_achievable_lhs),
            assumption5_violation=assumption5_violation,
        )

    if assumption5_violation:
        # Keep the least-violation/max-effort action, but do not relabel the
        # original calibrated constraint as feasible.
        u = maximizing_u
        feasible = False
    else:
        required = requested_xi - base
        if float(np.dot(a_u, u_nom)) >= required - tol:
            u = u_nom
        else:
            lo_lam, hi_lam = 0.0, 1.0
            def projected(lam):
                return np.clip(u_nom + lam * a_u, env.control_bounds.lo, env.control_bounds.hi)
            while float(np.dot(a_u, projected(hi_lam))) < required and hi_lam < 1e12:
                hi_lam *= 2.0
            for _ in range(80):
                mid = 0.5 * (lo_lam + hi_lam)
                if float(np.dot(a_u, projected(mid))) >= required:
                    hi_lam = mid
                else:
                    lo_lam = mid
            u = projected(hi_lam)
        feasible = True

    achieved_lhs = float(base + np.dot(a_u, u))
    margin = float(achieved_lhs - requested_xi)
    return FilterSolution(
        u=np.asarray(u, dtype=float),
        feasible=bool(feasible),
        margin=float(margin),
        B=terms.B,
        DtB=terms.DtB,
        a_u=terms.a_u,
        base_without_control=terms.base_without_control,
        requested_xi=requested_xi,
        achieved_lhs=achieved_lhs,
        max_achievable_lhs=float(max_achievable_lhs),
        assumption5_violation=assumption5_violation,
    )


# ===========================================================================
# Mismatch score η  (Eq. 4 in the paper)
# ===========================================================================

def eta_value(
    s: np.ndarray,
    u: np.ndarray,
    t: float,
    cbvf: CBVFTable,
    env: DubinsCBVFEnv,
) -> float:
    """
    η(s, u, t) = [ min_{d∈D} <∇B, f_app(x,u,d)>  −  <∇B, f(x,u,d)> ]₊

    Since there is no explicit disturbance d in this Dubins model, both
    terms simplify to the directional derivative at the applied control.
    """
    _require_disturbance_free_benchmark(env, "eta_value")
    _, grad, _ = cbvf.value_grad_dt(s, t)
    val_model = model_inner_product(s=s, u=u, grad=grad, env=env)
    val_true = float(np.dot(grad, env.true_dynamics(s, u)))
    return max(0.0, float(val_model - val_true))


def mismatch_sensitivity(s: np.ndarray, t: float, cbvf: CBVFTable) -> float:
    """Predeclared model-only scale for normalized directional mismatch.

    It uses only the nominal CBVF gradient and state heading, never the true
    dynamics parameters.  Split conformal is applied to eta/rho, and the
    deployed buffer is q*rho.  The small floor is used only in the score
    denominator; deployment returns zero when rho is numerically zero.
    """
    _, grad, _ = cbvf.value_grad_dt(s, t)
    th = float(s[2])
    return float(abs(grad[0] * np.cos(th) + grad[1] * np.sin(th)) + abs(grad[2]))


def trajectory_score(
    actions: List[float],
    states: List[np.ndarray],
    next_states: List[np.ndarray],
    times: List[float],
    next_times: List[float],
    cbvf: CBVFTable,
    env: DubinsCBVFEnv,
    partition: Optional[MondrianPartition] = None,
) -> Tuple[float, Dict[int, float], Dict[str, float]]:
    """
    Evaluate interval-tagged mismatch scores and form regional rollout maxima.

    For interval j, both ``(x_j, t_j^+)`` and ``(x_{j+1}, t_{j+1}^-)`` are
    evaluated using the held action ``u_j``.  Consequently an interior decision
    time is represented twice, once for the action to its left and once for the
    action to its right, exactly as in the paper's one-sided convention.

    This function returns raw realized regional maxima.  The grid margin is
    deliberately not added here: calibration adds it exactly once before
    quantile construction, while held-out and deployment exceedance checks
    compare raw realized mismatch against the already-inflated buffer.
    """
    lengths = {
        len(actions), len(states), len(next_states), len(times), len(next_times)
    }
    if len(lengths) != 1:
        raise ValueError("Interval-tagged trajectory arrays must have equal lengths.")
    max_eta = 0.0
    argmax_step = -1
    argmax_time = float(times[-1]) if len(times) > 0 else 0.0
    n_evals = 0
    regional_maxima: Dict[int, float] = {}
    for step_idx, (s_left, s_right, u, t_left, t_right) in enumerate(
        zip(states, next_states, actions, times, next_times)
    ):
        interval_points = (
            (s_left, float(t_left)),
            (s_right, float(t_right)),
        )
        for state, tagged_time in interval_points:
            eta_now = eta_value(state, u, tagged_time, cbvf, env)
            rho_now = mismatch_sensitivity(state, tagged_time, cbvf)
            normalized_eta = float(eta_now / max(rho_now, 1e-12))
            n_evals += 1
            if eta_now >= max_eta:
                max_eta = float(eta_now)
                argmax_step = int(step_idx)
                argmax_time = float(tagged_time)
            if partition is not None:
                region = partition.base_region(state)
                regional_maxima[region] = max(
                    regional_maxima.get(region, -float("inf")),
                    normalized_eta,
                )

    regional_scores = {
        int(region): float(score) for region, score in regional_maxima.items()
    }
    n_steps = max(1, len(times))
    diagnostics = {
        "max_eta_step": float(argmax_step),
        "max_eta_time": float(argmax_time),
        "max_eta_step_frac": float(max(0, argmax_step) / n_steps),
        "max_eta_on_last_step": float(1.0 if argmax_step == len(times) - 1 and len(times) > 0 else 0.0),
        "score_evals": float(n_evals),
        "visited_base_regions": float(len(regional_scores)),
    }
    return float(max_eta), regional_scores, diagnostics


# ===========================================================================
# Finite-sample conformal quantile used within every Mondrian region
# ===========================================================================

def _conformal_rank(n_scores: int, delta: float) -> int:
    """One-based finite-sample rank ceil((N+1)(1-delta))."""
    if not 0.0 < float(delta) < 1.0:
        raise ValueError("Conformal delta must lie strictly between zero and one.")
    if int(n_scores) < 1:
        raise ValueError("A conformal rank requires at least one score.")
    return int(math.ceil((int(n_scores) + 1) * (1.0 - float(delta))))


def _minimum_conformal_count(delta: float) -> int:
    """Smallest positive N whose finite-sample conformal rank is at most N.

    The initial closed-form candidate is corrected using ``_conformal_rank``
    itself. Thus the sparse-cell gate and ``conformal_quantile`` use exactly
    the same floating-point arithmetic and cannot disagree at a boundary.
    """
    if not 0.0 < float(delta) < 1.0:
        raise ValueError("Conformal delta must lie strictly between zero and one.")
    candidate = max(
        1,
        int(math.ceil(1.0 / float(delta) - 1e-12) - 1),
    )
    while candidate > 1 and _conformal_rank(candidate - 1, delta) <= candidate - 1:
        candidate -= 1
    while _conformal_rank(candidate, delta) > candidate:
        candidate += 1
    return int(candidate)


def conformal_quantile(scores: List[float], delta: float) -> float:
    """
    Return z_(k), where k = ceil((N+1)(1-delta)).

    Unlike the previous global implementation, this function does not clip an
    invalid rank to the sample maximum.  The exact finite-sample construction
    requires delta >= 1/(N+1); sparse Mondrian cells must be handled by the
    predeclared parent fallback instead.
    """
    if not 0.0 < float(delta) < 1.0:
        raise ValueError("Conformal delta must lie strictly between zero and one.")
    arr = np.sort(np.asarray(scores, dtype=float))
    N = len(arr)
    if N == 0:
        raise ValueError("Cannot calibrate a conformal quantile from zero scores.")
    if not np.isfinite(arr).all():
        raise ValueError("Conformal scores must all be finite.")
    minimum_n = _minimum_conformal_count(delta)
    if N < minimum_n:
        raise ValueError(
            f"N={N} is too small for delta={delta:.6g}; at least {minimum_n} "
            "regional rollout scores are required."
        )
    k_one_based = _conformal_rank(N, delta)
    if k_one_based > N:
        raise RuntimeError(
            "Internal conformal count invariant failed after the shared count guard."
        )
    return float(arr[k_one_based - 1])


# ===========================================================================
# Rollout engine
# ===========================================================================


def rollout(
    env: DubinsCBVFEnv,
    cbvf: CBVFTable,
    policy,
    tau: np.ndarray,
    init_state: np.ndarray,
    method: str,
    gamma: float,
    xi: float,
    shield_cfg: ShieldConfig,
    terminate_on_unsafe: bool = True,
    mondrian_calibration: Optional[MondrianCalibration] = None,
    score_partition: Optional[MondrianPartition] = None,
    speed_envelope: float = 0.0,
    epsilon_inter: float = 0.0,
) -> Dict[str, object]:
    """
    Run one episode. ``tau`` is a strictly increasing deployment-time
    subinterval contained in the stored CBVF horizon. The rollout starts at
    ``tau[0]`` and advances by ``env.dt`` per step. It may stop before the
    CBVF terminal slice when a positive terminal guard is configured.

    For evaluation, terminate_on_unsafe=True matches the benchmark.
    For calibration diagnostics, terminate_on_unsafe=False can be used so the
    conformal score is collected over a fuller trajectory.

    ``mondrian_calibration`` changes only the CP right-hand side.  The CBVF-QP
    objective, affine constraint, actuator bounds, activation logic, and exact
    projection solver remain unchanged.
    """
    if epsilon_inter < 0.0:
        raise ValueError("epsilon_inter must be nonnegative.")
    if speed_envelope < 0.0:
        raise ValueError("speed_envelope must be nonnegative.")
    tau = np.asarray(tau, dtype=float)
    if tau.ndim != 1 or len(tau) < 2:
        raise ValueError("tau must be a one-dimensional time axis with at least two points.")
    if not np.isfinite(tau).all() or not np.all(np.diff(tau) > 0.0):
        raise ValueError("tau must contain finite, strictly increasing physical times.")
    if len(tau) != int(env.horizon) + 1:
        raise ValueError(
            "Rollout horizon and CBVF time axis are misaligned: "
            f"env.horizon={env.horizon}, but len(tau)-1={len(tau) - 1}."
        )
    if not np.allclose(np.diff(tau), float(env.dt), rtol=1e-10, atol=1e-12):
        raise ValueError("The spacing of tau must match env.dt at every rollout step.")
    if len(cbvf.tau) > 1 and (
        float(tau[0]) < float(cbvf.tau[0]) - 1e-12
        or float(tau[-1]) > float(cbvf.tau[-1]) + 1e-12
    ):
        raise ValueError(
            "The deployment time axis must lie inside the stored CBVF time axis."
        )
    if score_partition is None and mondrian_calibration is not None:
        score_partition = mondrian_calibration.partition

    env.set_state(init_state)
    s = env.state.copy()

    step_idx = 0
    interventions = 0
    infeasible_steps = 0
    assumption5_violation_steps = 0
    constraint_violating_steps = 0
    active_filter_steps = 0
    early_activation_steps = 0
    max_eta_online = 0.0
    min_B = float("inf")
    min_l = float("inf")
    done = False
    cumulative_return = 0.0
    ever_unsafe = False
    ever_goal = False
    first_unsafe_step = -1
    is_shielding = False
    applied_buffer_values: List[float] = []
    applied_rhs_values: List[float] = []
    candidate_effective_region_counts: List[int] = []

    # Store interval-tagged trajectories for one-sided endpoint score evaluation.
    traj_states: List[np.ndarray] = []
    traj_next_states: List[np.ndarray] = []
    traj_actions: List[np.ndarray] = []
    traj_times: List[float] = []
    traj_next_times: List[float] = []

    while not done:
        if step_idx >= int(env.horizon):
            raise RuntimeError(
                "The rollout attempted to advance beyond the finite CBVF time axis."
            )
        t_now = float(tau[step_idx])

        u_nom = policy(s)
        B_pre, _, _ = cbvf.value_grad_dt(s, t_now)
        regional_buffer = float(xi)
        if method == "cp" and mondrian_calibration is not None:
            regional_coefficient, active_effective_regions = mondrian_calibration.applied_buffer(
                state=s,
                travel_radius=float(speed_envelope) * float(env.dt),
            )
            regional_buffer = float(regional_coefficient) * mismatch_sensitivity(
                s, t_now, cbvf
            )
            candidate_effective_region_counts.append(len(active_effective_regions))

        if method == "nominal":
            u = u_nom
            solution = None
            activation_threshold = -float("inf")
            qp_rhs = 0.0
            shield_active = False
        elif method in ["cbvf", "cp"]:
            activation_threshold = shield_cfg.activation_threshold(method, regional_buffer)
            qp_rhs = shield_cfg.qp_rhs(method, regional_buffer)
            if method == "cp":
                qp_rhs += float(epsilon_inter)
                applied_buffer_values.append(float(regional_buffer))
                applied_rhs_values.append(float(qp_rhs))

            # Hysteresis (anti-chattering): once the shield turns on,
            # keep it active until the trajectory has moved a little farther
            # away from the activation boundary.
            release_buffer = 0.05 if is_shielding else 0.0
            current_threshold = activation_threshold + release_buffer

            shield_active = bool(B_pre <= current_threshold)
            is_shielding = shield_active
            solution = None
            u = u_nom
            if shield_active:
                candidate = solve_cbvf_projection_exact(
                    s=s, t=t_now, u_nom=u_nom, cbvf=cbvf,
                    env=env, gamma=gamma, xi=qp_rhs,
                )
                u = candidate.u
                solution = candidate
        else:
            raise ValueError(f"Unknown method: {method}")

        if method in ["cbvf", "cp"] and shield_active:
            active_filter_steps += 1
            if B_pre > 0.0:
                early_activation_steps += 1
        if np.linalg.norm(np.asarray(u) - np.asarray(u_nom)) > 1e-10:
            interventions += 1

        if solution is not None:
            infeasible_steps += int(not solution.feasible)
            assumption5_violation_steps += int(solution.assumption5_violation)
            constraint_violating_steps += int(solution.margin < -1e-9)
            B_now = solution.B
        else:
            B_now = B_pre

        min_B = min(min_B, float(B_now))
        min_l = min(min_l, env.safety_margin_value(s))

        traj_states.append(s.copy())
        traj_actions.append(np.asarray(u, dtype=float).copy())
        traj_times.append(t_now)

        # Reward: negative distance to goal  + bonus on arrival
        dist_to_goal = goal_distance(env.world, s)
        reward = -dist_to_goal
        cumulative_return += reward

        s, step_done, step_info = env.step(u=u, terminate_on_unsafe=terminate_on_unsafe)
        step_idx += 1
        if step_idx >= len(tau):
            raise RuntimeError("Environment step count exceeded the validated CBVF time axis.")
        traj_next_states.append(s.copy())
        traj_next_times.append(float(tau[step_idx]))
        ever_unsafe = ever_unsafe or bool(step_info["unsafe"])
        ever_goal = ever_goal or bool(step_info["goal"])
        if bool(step_info["unsafe"]) and first_unsafe_step < 0:
            first_unsafe_step = step_idx
        done = bool(step_done)
        if not done and step_idx >= int(env.horizon):
            raise RuntimeError(
                "The environment did not terminate at its declared finite horizon."
            )

    # Final state statistics and full trajectory score.
    min_l = min(min_l, env.safety_margin_value(s))
    max_eta, regional_scores, eta_diag = trajectory_score(
        actions=traj_actions,
        states=traj_states,
        next_states=traj_next_states,
        times=traj_times,
        next_times=traj_next_times,
        cbvf=cbvf,
        env=env,
        partition=score_partition,
    )
    max_eta_online = max(max_eta_online, max_eta)
    effective_regional_scores: Dict[int, float] = {}
    buffer_exceeded = float("nan")
    buffer_coverage_available = 0.0
    if mondrian_calibration is not None:
        effective_regional_scores = mondrian_calibration.effective_rollout_scores(
            regional_scores
        )
        buffer_exceeded = float(
            not mondrian_calibration.rollout_is_covered(regional_scores)
        )
        buffer_coverage_available = 1.0

    return {
        "unsafe": float(ever_unsafe),
        "goal": float(ever_goal),
        "max_eta": float(max_eta_online),
        "max_eta_step": float(eta_diag["max_eta_step"]),
        "max_eta_time": float(eta_diag["max_eta_time"]),
        "max_eta_step_frac": float(eta_diag["max_eta_step_frac"]),
        "max_eta_on_last_step": float(eta_diag["max_eta_on_last_step"]),
        "min_B": float(min_B),
        "min_l": float(min_l),
        "interventions": float(interventions),
        "infeasible_steps": float(infeasible_steps),
        "assumption5_violation_steps": float(assumption5_violation_steps),
        "assumption5_violated": float(assumption5_violation_steps > 0),
        "constraint_violating_steps": float(constraint_violating_steps),
        "active_filter_steps": float(active_filter_steps),
        "early_activation_steps": float(early_activation_steps),
        "buffer_exceeded": float(buffer_exceeded),
        "buffer_coverage_available": float(buffer_coverage_available),
        "mean_applied_regional_buffer": float(np.mean(applied_buffer_values)) if applied_buffer_values else 0.0,
        "max_applied_regional_buffer": float(np.max(applied_buffer_values)) if applied_buffer_values else 0.0,
        "mean_applied_qp_rhs": float(np.mean(applied_rhs_values)) if applied_rhs_values else 0.0,
        "max_applied_qp_rhs": float(np.max(applied_rhs_values)) if applied_rhs_values else 0.0,
        "mean_candidate_effective_regions": (
            float(np.mean(candidate_effective_region_counts))
            if candidate_effective_region_counts else 0.0
        ),
        "max_candidate_effective_regions": (
            float(np.max(candidate_effective_region_counts))
            if candidate_effective_region_counts else 0.0
        ),
        "steps": float(step_idx),
        "first_unsafe_step": float(first_unsafe_step),
        "return": float(cumulative_return),
        "_traj_states": traj_states,
        "_traj_next_states": traj_next_states,
        "_traj_actions": traj_actions,
        "_traj_times": traj_times,
        "_traj_next_times": traj_next_times,
        "_regional_scores_raw": regional_scores,
        "_effective_regional_scores": effective_regional_scores,
    }

def summarize(logs: List[Dict[str, object]]) -> Dict[str, float]:
    scored_buffer_events = [
        float(x["buffer_exceeded"])
        for x in logs
        if float(x.get("buffer_coverage_available", 0.0)) > 0.5
    ]
    buffer_exceedance_rate = (
        float(np.mean(scored_buffer_events))
        if scored_buffer_events else float("nan")
    )
    return {
        "unsafe_rate": float(np.mean([x["unsafe"] for x in logs])),
        "safe_rate": float(1.0 - np.mean([x["unsafe"] for x in logs])),
        "goal_rate": float(np.mean([x["goal"] for x in logs])),
        "mean_return": float(np.mean([x["return"] for x in logs])),
        "total_unsafe_episodes": float(np.sum([x["unsafe"] for x in logs])),
        "mean_max_eta": float(np.mean([x["max_eta"] for x in logs])),
        "mean_max_eta_step_frac": float(np.mean([x["max_eta_step_frac"] for x in logs])),
        "frac_max_eta_on_last_step": float(np.mean([x["max_eta_on_last_step"] for x in logs])),
        "mean_min_B": float(np.mean([x["min_B"] for x in logs])),
        "mean_min_l": float(np.mean([x["min_l"] for x in logs])),
        "mean_interventions": float(np.mean([x["interventions"] for x in logs])),
        "mean_infeasible_steps": float(np.mean([x["infeasible_steps"] for x in logs])),
        "mean_assumption5_violation_steps": float(
            np.mean([x["assumption5_violation_steps"] for x in logs])
        ),
        "assumption5_violation_episode_rate": float(
            np.mean([x["assumption5_violated"] for x in logs])
        ),
        "mean_constraint_violating_steps": float(
            np.mean([x["constraint_violating_steps"] for x in logs])
        ),
        "mean_active_filter_steps": float(
            np.mean([x["active_filter_steps"] for x in logs])
        ),
        "mean_early_activation_steps": float(
            np.mean([x["early_activation_steps"] for x in logs])
        ),
        "buffer_exceedance_rate": buffer_exceedance_rate,
        "simultaneous_buffer_coverage": (
            float(1.0 - buffer_exceedance_rate)
            if np.isfinite(buffer_exceedance_rate) else float("nan")
        ),
        "n_buffer_scored_rollouts": float(len(scored_buffer_events)),
        "mean_applied_regional_buffer": float(
            np.mean([x["mean_applied_regional_buffer"] for x in logs])
        ),
        "mean_applied_qp_rhs": float(
            np.mean([x["mean_applied_qp_rhs"] for x in logs])
        ),
        "mean_candidate_effective_regions": float(
            np.mean([x["mean_candidate_effective_regions"] for x in logs])
        ),
        "max_candidate_effective_regions": float(
            np.max([x["max_candidate_effective_regions"] for x in logs])
        ),
        "mean_steps": float(np.mean([x["steps"] for x in logs])),
    }


def aggregate_method_stats(seed_stats: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate a method's per-seed summaries as mean ± std across seeds."""
    if not seed_stats:
        return {}
    keys = sorted({k for d in seed_stats for k in d.keys()})
    out: Dict[str, float] = {"n_seeds": float(len(seed_stats))}
    for key in keys:
        vals = []
        for d in seed_stats:
            v = d.get(key, None)
            if isinstance(v, (int, float, np.floating)) and np.isfinite(float(v)):
                vals.append(float(v))
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        out[key] = float(np.mean(arr))
        out[f"{key}_std_across_seeds"] = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return out


def aggregate_variant_seed_results(
    seed_results: List[Dict[str, Dict[str, float]]]
) -> Dict[str, Dict[str, float]]:
    """Aggregate all methods for one mismatch variant across seeds."""
    methods = ["nominal", "cbvf", "cp"]
    return {
        method: aggregate_method_stats([r[method] for r in seed_results if method in r])
        for method in methods
    }


def parse_seed_list(seed_text: Optional[str], fallback_seed: int) -> List[int]:
    """Parse comma/space-separated seeds while keeping --seed backward compatible."""
    if seed_text is None or str(seed_text).strip() == "":
        return [int(fallback_seed)]
    raw = str(seed_text).replace(";", ",").replace(" ", ",")
    seeds = [int(x) for x in raw.split(",") if x.strip() != ""]
    if not seeds:
        return [int(fallback_seed)]
    return seeds


def parse_strictly_increasing_floats(text: str, argument_name: str) -> Tuple[float, ...]:
    raw = str(text).replace(";", ",").replace(" ", ",")
    values = tuple(float(v) for v in raw.split(",") if v.strip() != "")
    if not values:
        raise ValueError(f"{argument_name} must contain at least one numeric edge.")
    if any(not np.isfinite(v) for v in values):
        raise ValueError(f"{argument_name} edges must be finite.")
    if any(values[i] >= values[i + 1] for i in range(len(values) - 1)):
        raise ValueError(f"{argument_name} edges must be strictly increasing.")
    return values


def parse_variant_selection(selection_text: Optional[str]) -> List[str]:
    """Parse requested deployment variants with user-friendly aliases.

    The experimental protocol in the paper now uses matched training and
    matched deployment. Therefore, each script invocation should usually
    evaluate only one variant with its corresponding RL checkpoint.
    """
    alias_to_name = {
        "all": "ALL",
        "aligned": "AlignedModel",
        "alignedmodel": "AlignedModel",
        "no_mismatch": "AlignedModel",
        "no_model_mismatch": "AlignedModel",
        "nomismatch": "AlignedModel",
        "no-model-mismatch": "AlignedModel",
        "matched": "AlignedModel",
        "hard": "HardMismatch",
        "hard_mismatch": "HardMismatch",
        "hardmismatch": "HardMismatch",
        "bad": "HardMismatch",
        "bad_mismatch": "HardMismatch",
        "easy": "EasyMismatch",
        "easy_mismatch": "EasyMismatch",
        "easymismatch": "EasyMismatch",
        "good": "EasyMismatch",
        "good_mismatch": "EasyMismatch",
    }
    if selection_text is None or str(selection_text).strip() == "":
        tokens = ["aligned"]
    else:
        raw = str(selection_text).replace(";", ",").replace(" ", ",")
        tokens = [tok.strip().lower() for tok in raw.split(",") if tok.strip()]
    selected: List[str] = []
    for tok in tokens:
        key = tok.replace("-", "_")
        # Keep a second lookup for aliases that intentionally contain hyphens.
        mapped = alias_to_name.get(key, alias_to_name.get(tok))
        if mapped is None:
            valid = ", ".join(sorted(alias_to_name.keys()))
            raise ValueError(f"Unknown --eval-variants entry '{tok}'. Valid aliases include: {valid}")
        if mapped == "ALL":
            return ["AlignedModel", "HardMismatch", "EasyMismatch"]
        if mapped not in selected:
            selected.append(mapped)
    return selected


def fmt_mean_std(stats: Dict[str, float], metric: str, digits: int = 3) -> str:
    mean = float(stats.get(metric, 0.0))
    std = float(stats.get(f"{metric}_std_across_seeds", 0.0))
    return f"{mean:.{digits}f}±{std:.{digits}f}"



# ===========================================================================
# Initial-state sampling
# ===========================================================================

def make_shared_initial_states(
    world: World,
    control_bounds: Interval,
    variant: DynamicsVariant,
    n_states: int,
    seed: int,
    dt: float,
    horizon: int,
    fixed_start: bool,
    start_jitter_xy: float,
    start_jitter_theta: float,
    cbvf: CBVFTable,
    tau: np.ndarray,
) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    env = DubinsCBVFEnv(
        world=world, control_bounds=control_bounds,
        variant=variant, dt=dt, horizon=horizon,
    )
    t0 = float(tau[0])
    if not fixed_start:
        return [env.sample_initial_state(rng, cbvf, t0) for _ in range(n_states)]

    x0 = np.array([
        world.init_x_range[0],
        world.init_y_range[0],
        world.init_theta_range[0],
    ], dtype=float)
    states: List[np.ndarray] = []
    for _ in range(n_states):
        candidate = x0.copy()
        if start_jitter_xy > 0.0:
            candidate[0] += rng.uniform(-start_jitter_xy, start_jitter_xy)
            candidate[1] += rng.uniform(-start_jitter_xy, start_jitter_xy)
        if start_jitter_theta > 0.0:
            candidate[2] = wrap_angle(
                candidate[2] + rng.uniform(-start_jitter_theta, start_jitter_theta)
            )
        if not env.is_safe_state(candidate):
            raise RuntimeError(
                "Requested fixed-start benchmark produced an unsafe initial state. "
                "Reduce the start jitter or change the default start."
            )
        B_cand, _, _ = cbvf.value_grad_dt(candidate, t0)
        if B_cand < 0.0:
            raise RuntimeError(
                "Jitter produced a state outside the certified safe set (B < 0). "
                "Reduce the jitter or change the default start."
            )
        states.append(candidate)
    return states


def assert_disjoint_initial_state_arrays(
    calibration_states: Sequence[np.ndarray],
    evaluation_states: Sequence[np.ndarray],
) -> Dict[str, object]:
    """Prevent literal calibration-state reuse in the held-out evaluation set.

    Independent draws support the exchangeability assumption.  This separate
    array-level check addresses data leakage: it verifies that no realized
    initial-state value appears in both sets.  In particular, it fails in the
    zero-jitter degenerate-start regime even when different RNG seeds are used.
    """
    if not calibration_states or not evaluation_states:
        raise ValueError("Calibration and evaluation state sets must both be nonempty.")
    calibration_array = np.stack(
        [np.asarray(state, dtype=float) for state in calibration_states], axis=0
    )
    evaluation_array = np.stack(
        [np.asarray(state, dtype=float) for state in evaluation_states], axis=0
    )
    if calibration_array.shape[1:] != evaluation_array.shape[1:]:
        raise ValueError("Calibration and evaluation states have incompatible shapes.")

    exact_overlap = np.all(
        calibration_array[:, np.newaxis, :] == evaluation_array[np.newaxis, :, :],
        axis=2,
    )
    overlap_pairs = np.argwhere(exact_overlap)
    if len(overlap_pairs) > 0:
        first_calib, first_eval = overlap_pairs[0]
        raise RuntimeError(
            "Held-out split violation: calibration and evaluation contain "
            f"{len(overlap_pairs)} identical initial-state pair(s). First overlap "
            f"is calibration[{int(first_calib)}] and evaluation[{int(first_eval)}]. "
            "Use nondegenerate start sampling; in particular, do not set both "
            "start jitters to zero for the coverage experiment."
        )
    return {
        "array_level_check": "exact_state_value_disjointness",
        "n_calibration_states": int(len(calibration_states)),
        "n_evaluation_states": int(len(evaluation_states)),
        "overlap_pairs": 0,
        "exchangeability_basis": "separate draws from the same fixed reference-policy protocol",
        "disjointness_purpose": "prevent calibration-to-evaluation data reuse",
    }


def exact_base_start(world: World) -> np.ndarray:
    return np.array([
        world.init_x_range[0],
        world.init_y_range[0],
        world.init_theta_range[0],
    ], dtype=float)


def start_certification_stats(cbvf: CBVFTable, tau: np.ndarray, states: List[np.ndarray]) -> Dict[str, float]:
    t0 = float(tau[0])
    b_vals = np.asarray([cbvf.value_grad_dt(s, t0)[0] for s in states], dtype=float)
    r = 0.50 + 0.20
    l_vals = np.asarray([
        float(np.sqrt((s[0] - 0.0) ** 2 + (s[1] - 0.0) ** 2) - r)
        for s in states
    ], dtype=float)
    return {
        "start_B_min": float(np.min(b_vals)),
        "start_B_mean": float(np.mean(b_vals)),
        "start_B_max": float(np.max(b_vals)),
        "start_B_positive_frac": float(np.mean(b_vals >= 0.0)),
        "start_l_min": float(np.min(l_vals)),
    }


def print_table_diagnostics(cbvf: CBVFTable, label: str) -> None:
    conv = cbvf.convergence_stats(n_tail_slices=5)
    if conv.get("time_invariant", 0.0) == 1.0:
        print(f"  table diagnostics [{label}]: converged (single-slice, time-invariant). DtB=0 everywhere.")
        return
    print(
        f"  table diagnostics [{label}]: "
        f"max|B(t0)-B(t0+dt)|={conv['earliest_slice_abs_diff']:.6f}, "
        f"rel={conv['earliest_slice_rel_diff']:.6f}, "
        f"tail_mean={conv['tail_slice_abs_diff_mean']:.6f}"
    )
    if conv['earliest_slice_rel_diff'] > 0.05:
        print("    WARNING: earliest CBVF slices are not yet close; the lookback horizon may be too short.")


def print_start_diagnostics(
    world: World,
    cbvf: CBVFTable,
    tau: np.ndarray,
    calib_states: List[np.ndarray],
    eval_states: List[np.ndarray],
) -> None:
    t0 = float(tau[0])
    x0 = exact_base_start(world)
    B_x0, _, _ = cbvf.value_grad_dt(x0, t0)
    print(f"  exact base start B(x0,t0) at x0=[-2,-2,pi/4]: {B_x0:.6f}")
    if B_x0 < 0.0:
        print("    WARNING: the exact Lu-style base start is outside the certified CBVF safe set.")
    calib = start_certification_stats(cbvf, tau, calib_states)
    evals = start_certification_stats(cbvf, tau, eval_states)
    print(
        f"  calibration starts: B_min={calib['start_B_min']:.6f}, "
        f"B_mean={calib['start_B_mean']:.6f}, B_max={calib['start_B_max']:.6f}, "
        f"positive_frac={calib['start_B_positive_frac']:.3f}"
    )
    print(
        f"  evaluation starts : B_min={evals['start_B_min']:.6f}, "
        f"B_mean={evals['start_B_mean']:.6f}, B_max={evals['start_B_max']:.6f}, "
        f"positive_frac={evals['start_B_positive_frac']:.3f}"
    )


# ===========================================================================
# Mondrian calibration: regional rollout scores and fixed parent fallback
# ===========================================================================

def collect_calibration_scores(
    world: World,
    control_bounds: Interval,
    variant: DynamicsVariant,
    cbvf: CBVFTable,
    policy,
    tau: np.ndarray,
    init_states: List[np.ndarray],
    gamma: float,
    method: str = "cbvf",
    xi: float = 0.0,
    shield_cfg: Optional[ShieldConfig] = None,
    continue_after_unsafe: bool = True,
    partition: Optional[MondrianPartition] = None,
    epsilon_grid: float = 0.0,
) -> Tuple[List[Dict[int, float]], List[Dict[str, object]]]:
    """
    Collect one regional maximum score per rollout and visited base region.

    The data-generating controller is the uncalibrated CBVF-QP (xi=0), as in the
    paper.  Its policy and filter stay fixed throughout offline collection.
    """
    if partition is None:
        raise ValueError("Mondrian calibration requires a fixed partition.")
    env = DubinsCBVFEnv(
        world=world, control_bounds=control_bounds,
        variant=variant, dt=float(tau[1] - tau[0]),
        horizon=len(tau) - 1,
    )
    if shield_cfg is None:
        shield_cfg = ShieldConfig(
            cbvf_activate_margin=0.0,
            cp_activate_margin=0.10,
        )
    scores: List[Dict[int, float]] = []
    diagnostics: List[Dict[str, object]] = []
    for x0 in init_states:
        out = rollout(
            env=env, cbvf=cbvf, policy=policy,
            tau=tau, init_state=x0,
            method=method, gamma=gamma, xi=xi,
            shield_cfg=shield_cfg,
            terminate_on_unsafe=not continue_after_unsafe,
            score_partition=partition,
        )
        raw_regional_scores = dict(out["_regional_scores_raw"])
        calibration_regional_scores = {
            int(region): float(raw_score + epsilon_grid)
            for region, raw_score in raw_regional_scores.items()
        }
        scores.append(calibration_regional_scores)
        out["_calibration_regional_scores_with_epsilon_grid"] = calibration_regional_scores
        diagnostics.append(out)
    return scores, diagnostics


def _build_strict_mondrian_tree(
    n_base_regions: int,
) -> Tuple[Tuple[int, ...], Dict[Tuple[int, ...], Tuple[int, ...]]]:
    """Build a fixed balanced binary tree over contiguous radial base cells.

    Every pair of tree nodes is either disjoint or nested, so independently
    promoted branches can never create partially overlapping effective regions.
    """
    if n_base_regions < 1:
        raise ValueError("A Mondrian partition needs at least one base region.")
    root = tuple(range(int(n_base_regions)))
    parent_by_node: Dict[Tuple[int, ...], Tuple[int, ...]] = {}

    def register_subtree(node: Tuple[int, ...]) -> None:
        if len(node) <= 1:
            return
        midpoint = len(node) // 2
        left = tuple(node[:midpoint])
        right = tuple(node[midpoint:])
        parent_by_node[left] = node
        parent_by_node[right] = node
        register_subtree(left)
        register_subtree(right)

    register_subtree(root)
    return root, parent_by_node


def _validate_tree_cut(
    cut: Sequence[Tuple[int, ...]],
    n_base_regions: int,
) -> Tuple[Tuple[int, ...], ...]:
    ordered = tuple(sorted((tuple(node) for node in cut), key=lambda node: node[0]))
    flattened = [base for node in ordered for base in node]
    expected = list(range(int(n_base_regions)))
    if flattened != expected:
        raise RuntimeError(
            "Mondrian tree cut must contain every base region exactly once with no overlap."
        )
    return ordered


def _promote_sparse_tree_cut(
    cut: Sequence[Tuple[int, ...]],
    sparse_nodes: Sequence[Tuple[int, ...]],
    parent_by_node: Dict[Tuple[int, ...], Tuple[int, ...]],
    n_base_regions: int,
) -> Tuple[Tuple[Tuple[int, ...], ...], Tuple[Tuple[int, ...], ...]]:
    """Promote only sparse branches while preserving a valid strict-tree cut."""
    candidate_parents: set = set()
    for node in sparse_nodes:
        if node not in parent_by_node:
            raise RuntimeError(
                "The global Mondrian region is still too sparse for the requested risk budget."
            )
        candidate_parents.add(parent_by_node[node])

    # If two requested parents are nested, the ancestor absorbs the descendant.
    promoted_parents = tuple(sorted(
        (
            parent for parent in candidate_parents
            if not any(
                set(parent) < set(other)
                for other in candidate_parents
            )
        ),
        key=lambda node: node[0],
    ))
    new_cut = [
        node for node in cut
        if not any(set(node).issubset(set(parent)) for parent in promoted_parents)
    ]
    new_cut.extend(promoted_parents)
    validated = _validate_tree_cut(new_cut, n_base_regions)
    if tuple(cut) == validated:
        raise RuntimeError("Sparse-region promotion made no progress.")
    return validated, promoted_parents


def _allocate_uniform_regional_risk(delta_traj: float, M: int) -> Tuple[float, ...]:
    """Allocate one Bonferroni term per leaf in the effective tree cut."""
    if M < 1:
        raise ValueError("Risk allocation requires at least one effective region.")
    share = float(delta_traj) / float(M)
    budgets = [share for _ in range(M - 1)]
    budgets.append(float(delta_traj) - math.fsum(budgets))
    if math.fsum(budgets) > float(delta_traj):
        budgets[-1] = float(np.nextafter(budgets[-1], 0.0))
    if any(budget <= 0.0 for budget in budgets):
        raise RuntimeError("Floating-point risk allocation produced a nonpositive budget.")
    if math.fsum(budgets) > float(delta_traj):
        raise RuntimeError("Regional risk allocation exceeds delta_traj.")
    return tuple(float(v) for v in budgets)


def _scores_for_effective_group(
    rollout_scores: Sequence[Dict[int, float]],
    group: Sequence[int],
) -> List[float]:
    group_set = set(int(v) for v in group)
    scores: List[float] = []
    for regional_scores in rollout_scores:
        visited = [
            float(score)
            for base_region, score in regional_scores.items()
            if int(base_region) in group_set
        ]
        if visited:
            scores.append(float(max(visited)))
    return scores

def calibrate_mondrian(
    world: World,
    control_bounds: Interval,
    variant: DynamicsVariant,
    cbvf: CBVFTable,
    policy,
    tau: np.ndarray,
    init_states: List[np.ndarray],
    gamma: float,
    delta_traj: float,
    shield_cfg: ShieldConfig,
    continue_after_unsafe: bool,
    partition: MondrianPartition,
    epsilon_grid: float,
) -> Tuple[MondrianCalibration, Dict[str, object]]:
    """
    Offline Mondrian split-conformal calibration from the revised paper.

    1. Roll out the BASELINE CBVF shield (xi=0) on the calibration set.
    2. Form z_{i,m}=max eta+epsilon_grid for every visited base region.
    3. Promote only sparse branches through a fixed strict binary tree until
       every leaf in the resulting mixed-depth tree cut has enough scores.
    4. Allocate one Bonferroni term to each tree-cut leaf and calibrate one
       quantile per leaf. The regional budgets sum to at most delta_traj.
    """
    if not 0.0 < float(delta_traj) < 1.0:
        raise ValueError("delta_traj must lie strictly between zero and one.")
    if epsilon_grid < 0.0:
        raise ValueError("epsilon_grid must be nonnegative.")

    rollout_scores, rollout_diags = collect_calibration_scores(
        world=world,
        control_bounds=control_bounds,
        variant=variant,
        cbvf=cbvf,
        policy=policy,
        tau=tau,
        init_states=init_states,
        gamma=gamma,
        method="cbvf",
        xi=0.0,
        shield_cfg=shield_cfg,
        continue_after_unsafe=continue_after_unsafe,
        partition=partition,
        epsilon_grid=epsilon_grid,
    )

    tree_root, parent_by_node = _build_strict_mondrian_tree(partition.n_base_regions)
    predeclared_tree_nodes = set(parent_by_node)
    predeclared_tree_nodes.add(tree_root)
    selected_groups = _validate_tree_cut(
        tuple((base,) for base in range(partition.n_base_regions)),
        partition.n_base_regions,
    )
    promotion_history: List[Dict[str, object]] = []
    promotion_rounds = 0

    while True:
        deltas = _allocate_uniform_regional_risk(delta_traj, len(selected_groups))
        selected_group_scores = [
            _scores_for_effective_group(rollout_scores, group)
            for group in selected_groups
        ]
        counts_now = [len(values) for values in selected_group_scores]
        minimum_counts = [
            _minimum_conformal_count(delta_m) for delta_m in deltas
        ]
        sparse_nodes = tuple(
            group
            for group, count, minimum_n in zip(
                selected_groups, counts_now, minimum_counts
            )
            if count < minimum_n
        )
        if not sparse_nodes:
            break

        cut_before = selected_groups
        try:
            selected_groups, promoted_parents = _promote_sparse_tree_cut(
                cut=cut_before,
                sparse_nodes=sparse_nodes,
                parent_by_node=parent_by_node,
                n_base_regions=partition.n_base_regions,
            )
        except RuntimeError as exc:
            global_minimum = _minimum_conformal_count(float(delta_traj))
            raise RuntimeError(
                "The global Mondrian fallback is still too sparse: "
                f"n_calib={len(rollout_scores)}, but at least {global_minimum} "
                f"rollouts are required for delta_traj={delta_traj:.6g}."
            ) from exc
        promotion_history.append({
            "round": int(promotion_rounds),
            "cut_before": [[int(v) for v in group] for group in cut_before],
            "counts_before": [int(v) for v in counts_now],
            "minimum_counts_before": [int(v) for v in minimum_counts],
            "sparse_nodes": [[int(v) for v in group] for group in sparse_nodes],
            "promoted_parents": [
                [int(v) for v in group] for group in promoted_parents
            ],
            "cut_after": [[int(v) for v in group] for group in selected_groups],
        })
        promotion_rounds += 1

    # These are the leaves of the final mixed-depth tree cut. Allocate exactly
    # one Bonferroni term and calibrate exactly one buffer per cut leaf.
    M_effective = len(selected_groups)
    if any(group not in predeclared_tree_nodes for group in selected_groups):
        raise RuntimeError(
            "The effective Mondrian cut contains a group outside the predeclared region tree."
        )
    deltas = _allocate_uniform_regional_risk(delta_traj, M_effective)
    selected_group_scores = [
        _scores_for_effective_group(rollout_scores, group)
        for group in selected_groups
    ]
    buffers = tuple(
        conformal_quantile(values, delta_m)
        for values, delta_m in zip(selected_group_scores, deltas)
    )
    counts = tuple(len(values) for values in selected_group_scores)
    base_to_effective = [-1] * partition.n_base_regions
    for effective_region, group in enumerate(selected_groups):
        for base_region in group:
            base_to_effective[int(base_region)] = int(effective_region)
    if any(region < 0 for region in base_to_effective):
        raise RuntimeError("Sparse-region fallback left a base region unmapped.")

    calibration = MondrianCalibration(
        partition=partition,
        effective_groups=tuple(tuple(int(v) for v in group) for group in selected_groups),
        base_to_effective=tuple(int(v) for v in base_to_effective),
        buffers=tuple(float(v) for v in buffers),
        deltas=tuple(float(v) for v in deltas),
        counts=tuple(int(v) for v in counts),
        delta_traj=float(delta_traj),
        epsilon_grid=float(epsilon_grid),
        promotion_rounds=int(promotion_rounds),
    )

    flattened_scores = np.asarray(
        [score for regional in rollout_scores for score in regional.values()],
        dtype=float,
    )
    if len(flattened_scores) == 0:
        raise RuntimeError("Calibration rollouts did not visit any Mondrian region.")
    xi_max = calibration.max_buffer
    diag_cfg = shield_cfg.describe(float(xi_max))
    base_visit_counts = [
        len(_scores_for_effective_group(rollout_scores, (base_region,)))
        for base_region in range(partition.n_base_regions)
    ]
    stats = {
        "n_calib": float(len(rollout_scores)),
        "delta_traj": float(delta_traj),
        "n_base_regions": float(partition.n_base_regions),
        "n_effective_regions": float(calibration.n_effective_regions),
        "promotion_rounds": float(calibration.promotion_rounds),
        "risk_budget_sum": float(math.fsum(calibration.deltas)),
        "base_visit_counts": [int(v) for v in base_visit_counts],
        "zero_visit_base_regions": [
            int(region)
            for region, count in enumerate(base_visit_counts)
            if count == 0
        ],
        "epsilon_grid": float(epsilon_grid),
        "score_min": float(np.min(flattened_scores)),
        "score_mean": float(np.mean(flattened_scores)),
        "score_max": float(np.max(flattened_scores)),
        "score_p50": float(np.median(flattened_scores)),
        "score_p90": float(np.percentile(flattened_scores, 90)),
        "nonzero_frac": float((flattened_scores > 1e-10).mean()),
        # xi_hat remains as a compatibility summary and is the maximum regional buffer.
        "xi_hat": float(xi_max),
        "xi_hat_max": float(xi_max),
        "xi_hat_min": float(min(calibration.buffers)),
        "regions": calibration.as_dict()["regions"],
        "regional_normalized_score_samples": [
            [float(value) for value in values]
            for values in selected_group_scores
        ],
        "score_normalization": (
            "eta/(abs(grad_B_dot_heading)+abs(dB_dtheta)); "
            "deployed xi=q_region*(abs(grad_B_dot_heading)+abs(dB_dtheta))"
        ),
        "promotion_history": promotion_history,
        "cbvf_activate_margin": float(diag_cfg["cbvf_activate_margin"]),
        "cp_activate_margin": float(diag_cfg["cp_activate_margin"]),
        "cbvf_activation_threshold": float(diag_cfg["cbvf_activation_threshold"]),
        "cp_activation_threshold": float(diag_cfg["cp_activation_threshold"]),
        "cbvf_qp_rhs": float(diag_cfg["cbvf_qp_rhs"]),
        "cp_qp_rhs": float(diag_cfg["cp_qp_rhs"]),
        "mean_max_eta_step_frac": float(np.mean([d["max_eta_step_frac"] for d in rollout_diags])),
        "frac_max_eta_on_last_step": float(np.mean([d["max_eta_on_last_step"] for d in rollout_diags])),
        "mean_active_filter_steps": float(np.mean([d["active_filter_steps"] for d in rollout_diags])),
        "mean_early_activation_steps": float(np.mean([d["early_activation_steps"] for d in rollout_diags])),
        "mean_calib_unsafe": float(np.mean([d["unsafe"] for d in rollout_diags])),
        "continue_after_unsafe": float(1.0 if continue_after_unsafe else 0.0),
    }
    return calibration, stats


# ===========================================================================
# Evaluation
# ===========================================================================

def evaluate_method(
    world: World,
    control_bounds: Interval,
    variant: DynamicsVariant,
    cbvf: CBVFTable,
    policy,
    tau: np.ndarray,
    init_states: List[np.ndarray],
    method: str,
    xi: float,
    gamma: float,
    shield_cfg: ShieldConfig,
    mondrian_calibration: Optional[MondrianCalibration] = None,
    speed_envelope: float = 0.0,
    epsilon_inter: float = 0.0,
) -> List[Dict[str, object]]:
    env = DubinsCBVFEnv(
        world=world, control_bounds=control_bounds,
        variant=variant, dt=float(tau[1] - tau[0]),
        horizon=len(tau) - 1,
    )
    logs: List[Dict[str, object]] = []
    for x0 in init_states:
        logs.append(
            rollout(
                env=env, cbvf=cbvf, policy=policy,
                tau=tau, init_state=x0,
                method=method, gamma=gamma, xi=xi,
                shield_cfg=shield_cfg,
                mondrian_calibration=mondrian_calibration,
                score_partition=(
                    mondrian_calibration.partition
                    if mondrian_calibration is not None else None
                ),
                speed_envelope=speed_envelope,
                epsilon_inter=epsilon_inter,
            )
        )
    return logs


# ===========================================================================
# Printing helpers
# ===========================================================================

def print_calibration_stats(stats: Dict[str, object], variant_name: str = "") -> None:
    print(
        f"  Mondrian CP calibration: n={int(stats['n_calib'])}, "
        f"M_base={int(stats['n_base_regions'])}, "
        f"M_effective={int(stats['n_effective_regions'])}, "
        f"promotion_rounds={int(stats['promotion_rounds'])}, "
        f"sum_delta_m={stats['risk_budget_sum']:.6f}, "
        f"score_min={stats['score_min']:.6f}, "
        f"score_mean={stats['score_mean']:.6f}, "
        f"score_max={stats['score_max']:.6f}, "
        f"score_p90={stats['score_p90']:.6f}, "
        f"nonzero_frac={stats['nonzero_frac']:.3f}, "
        f"xi_range=[{stats['xi_hat_min']:.6f}, {stats['xi_hat_max']:.6f}]"
    )
    for region in stats["regions"]:
        labels = " OR ".join(region["base_region_labels"])
        print(
            f"    region {region['effective_region']}: n={region['n_scores']}, "
            f"delta_m={region['delta_m']:.6f}, "
            f"xi_hat_off={region['xi_hat_off']:.6f} | {labels}"
        )
    print(
        f"    cbvf_margin={stats['cbvf_activate_margin']:.4f}, "
        f"cp_margin={stats['cp_activate_margin']:.4f}, "
        f"cbvf_thresh={stats['cbvf_activation_threshold']:.4f}, "
        f"cp_thresh={stats['cp_activation_threshold']:.4f}, "
        f"cbvf_rhs={stats['cbvf_qp_rhs']:.6f}, "
        f"max_cp_rhs_before_epsilon_inter={stats['cp_qp_rhs']:.6f}, "
        f"epsilon_grid={stats['epsilon_grid']:.6f}"
    )
    print(
        f"    calib_mean_active={stats['mean_active_filter_steps']:.2f}, "
        f"calib_mean_early_active={stats['mean_early_activation_steps']:.2f}, "
        f"mean_eta_argmax_frac={stats['mean_max_eta_step_frac']:.3f}, "
        f"frac_eta_last_step={stats['frac_max_eta_on_last_step']:.3f}, "
        f"calib_unsafe_rate={stats['mean_calib_unsafe']:.3f}, "
        f"continue_after_unsafe={bool(stats['continue_after_unsafe'])}"
    )
    if stats['nonzero_frac'] < 0.10:
        if variant_name == "AlignedModel":
            print("    NOTE: regional eta scores are degenerate in AlignedModel, as expected; Mondrian CP collapses toward CBVF.")
        else:
            print("    WARNING: regional eta scores are nearly degenerate; CP and CBVF may become analytically identical.")
    if stats['frac_max_eta_on_last_step'] > 0.70:
        print("    WARNING: max eta is often achieved at the last rollout step; calibration may still be too late-horizon dominated.")


# ===========================================================================
# Plotting helpers
# ===========================================================================

def _finalize_plot(output_path: Path, show: bool) -> None:
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_grouped_bars(
    results: Dict[str, Dict[str, Dict[str, float]]],
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
    show: bool,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    variant_names = list(results.keys())
    methods = ["nominal", "cbvf", "cp"]
    colors = {"nominal": "#e15759", "cbvf": "#4e79a7", "cp": "#59a14f"}
    hatches = {"nominal": "", "cbvf": "//", "cp": ".."}
    x = np.arange(len(variant_names))
    width = 0.25

    plt.figure(figsize=(9, 4.5))
    for j, method in enumerate(methods):
        vals = [float(results[v][method].get(metric, 0.0)) for v in variant_names]
        errs = [float(results[v][method].get(f"{metric}_std_across_seeds", 0.0)) for v in variant_names]
        use_errs = any(e > 0.0 for e in errs)
        plt.bar(
            x + (j - 1) * width, vals, width=width,
            yerr=errs if use_errs else None, capsize=3 if use_errs else 0,
            label=method, color=colors[method], alpha=0.88,
            edgecolor="black", linewidth=0.9, hatch=hatches[method],
        )

    plt.xticks(x, variant_names, rotation=0)
    plt.ylabel(ylabel)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    elif metric.endswith("_rate"):
        plt.ylim(0.0, 1.0)
    plt.legend()
    _finalize_plot(output_path=output_path, show=show)


def plot_safety_comparison(
    results: Dict[str, Dict[str, Dict[str, float]]],
    output_path: Path,
    show: bool,
) -> None:
    """Side-by-side unsafe_rate and goal_rate for a clean safety overview."""
    variant_names = list(results.keys())
    methods = ["nominal", "cbvf", "cp"]
    colors = {"nominal": "#e15759", "cbvf": "#4e79a7", "cp": "#59a14f"}
    hatches = {"nominal": "", "cbvf": "//", "cp": ".."}
    x = np.arange(len(variant_names))
    width = 0.22

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (metric, ylabel) in zip(
        axes,
        [("unsafe_rate", "Unsafe episode rate"), ("goal_rate", "Goal-reaching rate")],
    ):
        for j, method in enumerate(methods):
            vals = [float(results[v][method].get(metric, 0.0)) for v in variant_names]
            errs = [float(results[v][method].get(f"{metric}_std_across_seeds", 0.0)) for v in variant_names]
            use_errs = any(e > 0.0 for e in errs)
            ax.bar(
                x + (j - 1) * width, vals, width=width,
                yerr=errs if use_errs else None, capsize=3 if use_errs else 0,
                label=method, color=colors[method], alpha=0.88,
                edgecolor="black", linewidth=0.9, hatch=hatches[method],
            )
        ax.set_xticks(x)
        ax.set_xticklabels(variant_names, rotation=0)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend()
        if metric.endswith("_rate"):
            ax.set_ylim(0.0, 1.0)

    fig.suptitle("Safety comparison: Nominal vs CBVF vs Conformal-CBVF shield")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_trajectories_for_variant(
    variant_name: str,
    eval_logs: Dict[str, list],
    world: "World",
    output_path: Path,
    show: bool,
    n_trajs: int = 10,
) -> None:
    """
    Plot representative trajectories for nominal, cbvf, and cp on one axes.
    eval_logs: dict mapping method -> list of rollout dicts (each with _traj_states).
    n_trajs: how many trajectories to draw per method.
    """
    colors = {"nominal": "#e15759", "cbvf": "#4e79a7", "cp": "#59a14f"}
    labels = {"nominal": "Nominal", "cbvf": "CBVF", "cp": "Conformal-CBVF"}
    methods = ["nominal", "cbvf", "cp"]

    fig, ax = plt.subplots(figsize=(5, 5))

    # obstacle (combined robot + obstacle radius)
    obs_cx, obs_cy = world.obstacle_center
    obs_r = world.obstacle_radius + world.robot_radius
    ax.add_patch(plt.Circle((obs_cx, obs_cy), obs_r, color="black", zorder=5))

    # goal region
    gx, gy = world.goal
    ax.add_patch(plt.Circle(
        (gx, gy), world.goal_radius,
        facecolor="#59a14f", alpha=0.25,
        edgecolor="#2d7a2d", linewidth=1.2, zorder=4,
    ))
    ax.plot(gx, gy, "+", color="#2d7a2d", markersize=10, zorder=6)

    start_plotted = False
    legend_handles = [
        plt.Patch(facecolor="#59a14f", alpha=0.3,
                  edgecolor="#2d7a2d", label="Goal"),
    ]

    for method in methods:
        rollouts = eval_logs.get(method, [])
        if not rollouts:
            continue
        plotted = 0
        for ro in rollouts:
            states = ro.get("_traj_states", [])
            if not states:
                continue
            xs = [s[0] for s in states]
            ys = [s[1] for s in states]
            ax.plot(xs, ys,
                    color=colors[method], linewidth=1.0,
                    alpha=0.70, zorder=3)
            if not start_plotted:
                ax.plot(xs[0], ys[0], "*",
                        color="gold", markersize=14, zorder=7,
                        markeredgecolor="black", markeredgewidth=0.5)
                start_plotted = True
            plotted += 1
            if plotted >= n_trajs:
                break
        legend_handles.append(
            plt.Line2D([0], [0], color=colors[method],
                       linewidth=1.5, label=labels[method])
        )

    legend_handles.append(
        plt.Line2D([0], [0], marker="*", color="gold",
                   linestyle="None", markersize=10,
                   markeredgecolor="black", markeredgewidth=0.5,
                   label="Start")
    )

    ax.set_xlim(world.xmin, world.xmax)
    ax.set_ylim(world.ymin, world.ymax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Representative trajectories: {variant_name}")
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    _finalize_plot(output_path=output_path, show=show)

def build_mismatch_variants(
    *,
    aligned_speed: float,
    matched_beta: float,
    hard_model_speed: float,
    hard_truth_speed: float,
    hard_model_beta: float,
    hard_truth_beta: float,
    easy_model_speed: float,
    easy_truth_speed: float,
    easy_model_beta: float,
    easy_truth_beta: float,
) -> List[DynamicsVariant]:
    """Build the three paper-facing mismatch scenarios."""
    return [
        DynamicsVariant(
            name="AlignedModel",
            model=DynamicsSpec(v=aligned_speed, beta_u=matched_beta),
            truth=DynamicsSpec(v=aligned_speed, beta_u=matched_beta),
        ),
        DynamicsVariant(
            name="HardMismatch",
            model=DynamicsSpec(v=hard_model_speed, beta_u=hard_model_beta),
            truth=DynamicsSpec(v=hard_truth_speed, beta_u=hard_truth_beta),
        ),
        DynamicsVariant(
            name="EasyMismatch",
            model=DynamicsSpec(v=easy_model_speed, beta_u=easy_model_beta),
            truth=DynamicsSpec(v=easy_truth_speed, beta_u=easy_truth_beta),
        ),
    ]


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Mondrian conformal CBVF shielding experiment "
            "(Dubins car, circular obstacle).\n"
            "Implements region-wise calibration under approximate dynamics."
        )
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help="Directory for CBVF .npz tables. Defaults to ./cbvf_cache next to this script.",
    )
    parser.add_argument(
        "--recompute-cbvf", dest="recompute_cbvf", action="store_true",
        help="Rebuild offline CBVF tables from scratch (the default).",
    )
    parser.add_argument(
        "--use-cached-cbvf", dest="recompute_cbvf", action="store_false",
        help="Explicitly opt into loading a compatible cached CBVF table.",
    )
    parser.set_defaults(recompute_cbvf=True)
    parser.add_argument("--cbvf-nx", type=int, default=81)
    parser.add_argument("--cbvf-ny", type=int, default=81)
    parser.add_argument("--cbvf-nth", type=int, default=61)
    parser.add_argument(
        "--fast", action="store_true",
        help="Quick-test mode: medium debugging grid (51x51x41), short horizon (100 steps), "
             "few rollouts (20 calib / 20 eval). Not for paper-quality conclusions.",
    )
    parser.add_argument(
        "--cbvf-dt", type=float, default=0.10,
        help="Time step used for the offline CBVF solve.",
    )
    parser.add_argument(
        "--solver-accuracy", type=str, default="low",
        choices=["low", "medium", "high"],
    )
    parser.add_argument("--n-calib", type=int, default=500)
    parser.add_argument("--n-eval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7, help="Backward-compatible single seed used only if --seeds is empty.")
    parser.add_argument("--seeds", type=str, default="7,42,123,0,99",
                        help="Comma-separated seeds for mean±std reporting, e.g. 7,42,123,0,99. Use --seeds 7 for one quick run.")
    parser.add_argument("--gamma", type=float, default=0.10)
    parser.add_argument("--delta-traj", "--delta", dest="delta_traj", type=float, default=0.05,
                        help="Trajectory-level Mondrian miscoverage budget. The alias --delta is retained. Default 0.05 gives 95%% reference-rollout coverage.")
    parser.add_argument(
        "--mondrian-clearance-edges", type=str, default="0.25,0.75,1.50",
        help=(
            "Strictly increasing obstacle-clearance edges for the fixed radial "
            "Mondrian partition. Three edges create four base regions."
        ),
    )
    parser.add_argument(
        "--epsilon-grid", type=float, default=0.0,
        help=(
            "Nonnegative inter-evaluation score margin added once to every "
            "regional rollout score (set to a validated L_eta*h_eval bound)."
        ),
    )
    parser.add_argument(
        "--epsilon-inter", type=float, default=0.0,
        help=(
            "Nonnegative sampled-control margin added to the active regional "
            "QP right-hand side (set to a validated L_Psi*(C_bar+1)*dt bound)."
        ),
    )
    parser.add_argument(
        "--speed-envelope", type=float, default=None,
        help=(
            "Known full-state speed envelope C_bar used to find every Mondrian "
            "region reachable in one control step. If omitted, the simulator "
            "uses a conservative envelope over its configured dynamics variants."
        ),
    )
    parser.add_argument("--dt", type=float, default=0.05,
                        help="Simulation step size.")
    parser.add_argument("--horizon", type=int, default=400,
                        help="Maximum rollout length in steps.")
    parser.add_argument(
        "--target-shape", type=str,
        choices=["signed_distance", "squared"],
        default="signed_distance",
    )
    parser.add_argument("--target-clip", type=float, default=1.0)
    parser.add_argument("--max-abs-table", type=float, default=10.0,
                        help="Max allowed magnitude in the saved CBVF table.")
    parser.add_argument("--max-abs-grad", type=float, default=200.0,
                        help="Max allowed gradient magnitude in the CBVF table.")
    parser.add_argument("--max-abs-solver-value", type=float, default=50.0,
                        help="Explosion guard on the VALUE FUNCTION only (not Hamiltonian).")
    parser.add_argument(
        "--max-reasonable-xi", type=float, default=5.0,
        help="Reject xi_hat above this value as numerically suspect.",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--plot-dir", type=str, default=None)
    parser.add_argument("--results-json", type=str, default=None,
                        help="Optional path for saving the numeric results and metadata as JSON.")
    parser.add_argument(
        "--eval-variants", type=str, default="aligned",
        help=(
            "Deployment variant(s) to evaluate: aligned/no_mismatch, hard/bad, "
            "easy/good, or all. Default is aligned only for the matched-policy "
            "protocol: use the aligned RL checkpoint only in the aligned environment."
        ),
    )
    parser.add_argument("--show-plots", action="store_true")
    parser.add_argument("--k-heading", type=float, default=3.0)
    parser.add_argument("--nominal-controller", choices=["rl", "heuristic"], default="rl",
                        help="Nominal controller used before the CBVF/CP shield. Default is the trained RL policy.")
    parser.add_argument("--rl-checkpoint-dir", type=str, default="rl_checkpoints/reach_avoid_nominal_challenging",
                        help="Directory containing params_<step>.pkl saved by train/train_sac_lag.py. Default is the aligned/no-mismatch nominal checkpoint for the matched-policy protocol.")
    parser.add_argument("--rl-checkpoint-step", type=str, default="latest",
                        help="Checkpoint step to load, or 'latest'.")
    parser.add_argument("--rl-hidden-dims", type=str, default="256,256",
                        help="Actor hidden dimensions used when no policy_config.json is available.")
    parser.add_argument("--rl-seed", type=int, default=42,
                        help="Seed used to initialize the actor structure before restoring the checkpoint.")
    parser.add_argument("--rl-stochastic-action", action="store_true",
                        help="Use stochastic actor samples instead of deterministic actor mode during rollouts.")
    parser.add_argument("--random-start", action="store_true",
                        help="Disable the Lu-style fixed-start benchmark and sample starts instead.")
    parser.add_argument("--start-jitter-xy", type=float, default=0.30,
                        help="Uniform jitter added to the Lu-style fixed x/y start. "
                             "Default 0.30 creates enough variance in B(x0) across rollouts "
                             "for both the baseline threshold and the calibrated CP threshold "
                             "to be meaningfully tested across different starts.")
    parser.add_argument("--start-jitter-theta", type=float, default=0.15,
                        help="Uniform jitter added to the Lu-style fixed heading. "
                             "Default 0.15 rad creates heading diversity that complements "
                             "the xy jitter in generating varied B(x0) values.")
    parser.add_argument("--activate-margin", type=float, default=None,
                        help="Backward-compatible shortcut. If provided, sets both --cbvf-activate-margin and --cp-activate-margin to this value.")
    parser.add_argument("--cbvf-activate-margin", type=float, default=0.0,
                        help="Baseline CBVF wake-up buffer. Default 0.0 makes CBVF least-restrictive: it activates only at B <= 0.")
    parser.add_argument("--cp-activate-margin", type=float, default=0.10,
                        help="Conformal-CBVF wake-up buffer. CP keeps a small wake-up margin and tightens the QP RHS with xi_hat.")
    parser.add_argument("--max-est-runtime-gb", type=float, default=3.0,
                        help="Skip offline tables whose estimated runtime footprint exceeds this many GiB.")
    parser.set_defaults(calib_continue_after_unsafe=True)
    parser.add_argument("--no-calib-continue-after-unsafe", dest="calib_continue_after_unsafe", action="store_false",
                        help="During calibration, stop immediately on the first safety violation.")
    parser.add_argument("--speed", type=float, default=0.60,
                        help="Default speed used by AlignedModel and, unless overridden, the mismatch variants.")
    parser.add_argument("--matched-beta", type=float, default=0.50)
    parser.add_argument("--hard-model-speed", type=float, default=None,
                        help="Model speed for HardMismatch. Defaults to --speed.")
    parser.add_argument("--hard-truth-speed", type=float, default=None,
                        help="True speed for HardMismatch. Defaults to --speed. Set lower than --hard-model-speed to add speed optimism.")
    parser.add_argument("--hard-model-beta", "--bad-model-beta", dest="hard_model_beta", type=float, default=0.90,
                        help="Model beta for HardMismatch. The alias --bad-model-beta is kept for backward compatibility.")
    parser.add_argument("--hard-truth-beta", "--bad-truth-beta", dest="hard_truth_beta", type=float, default=0.40,
                        help="True beta for HardMismatch. Default 0.40 versus model beta 0.90 makes the model strongly optimistic.")
    parser.add_argument("--easy-model-speed", type=float, default=None,
                        help="Model speed for EasyMismatch. Defaults to --speed.")
    parser.add_argument("--easy-truth-speed", type=float, default=None,
                        help="True speed for EasyMismatch. Defaults to --speed.")
    parser.add_argument("--easy-model-beta", "--good-model-beta", dest="easy_model_beta", type=float, default=0.50)
    parser.add_argument("--easy-truth-beta", "--good-truth-beta", dest="easy_truth_beta", type=float, default=0.75)
    parser.add_argument("--allow-coarse-fallback", action="store_true",
                        help="Allow coarser fallback grids if the requested high-resolution table fails to build.")
    parser.add_argument("--odp-root", type=str, default=None)
    parser.add_argument("--solver-file", type=str, default=None)
    args = parser.parse_args()
    if not 0.0 < float(args.delta_traj) < 1.0:
        parser.error("--delta-traj must lie strictly between zero and one.")
    if args.epsilon_grid < 0.0 or args.epsilon_inter < 0.0:
        parser.error("--epsilon-grid and --epsilon-inter must be nonnegative.")
    if args.speed_envelope is not None and args.speed_envelope <= 0.0:
        parser.error("--speed-envelope must be positive when provided.")
    if args.activate_margin is not None:
        args.cbvf_activate_margin = float(args.activate_margin)
        args.cp_activate_margin = float(args.activate_margin)
    args.time_invariant_cbvf = False
    args.target_scale = 0.0
    if args.hard_model_speed is None:
        args.hard_model_speed = float(args.speed)
    if args.hard_truth_speed is None:
        args.hard_truth_speed = float(args.speed)
    if args.easy_model_speed is None:
        args.easy_model_speed = float(args.speed)
    if args.easy_truth_speed is None:
        args.easy_truth_speed = float(args.speed)
    seeds = parse_seed_list(args.seeds, args.seed)

    # --fast: override grid / horizon / rollout count for quick testing
    if args.fast:
        args.cbvf_nx = 41
        args.cbvf_ny = 41
        args.cbvf_nth = 31
        args.horizon = 200
        args.n_calib = 20
        args.n_eval = 20
        args.cbvf_dt = 0.20
        print("[--fast] Quick-test mode: 41×41×31 grid, 200-step horizon, 20 rollouts. Not for paper-quality conclusions.\n")

    args.fixed_start = not args.random_start


    random.seed(seeds[0])
    np.random.seed(seeds[0])

    world = World()
    control_bounds = Interval(lo=-1.0, hi=1.0)
    mondrian_partition = MondrianPartition(
        clearance_edges=parse_strictly_increasing_floats(
            args.mondrian_clearance_edges,
            "--mondrian-clearance-edges",
        ),
        obstacle_center=world.obstacle_center,
        inflated_obstacle_radius=float(world.obstacle_radius + world.robot_radius),
    )
    variants = build_mismatch_variants(
        aligned_speed=args.speed,
        matched_beta=args.matched_beta,
        hard_model_speed=args.hard_model_speed,
        hard_truth_speed=args.hard_truth_speed,
        hard_model_beta=args.hard_model_beta,
        hard_truth_beta=args.hard_truth_beta,
        easy_model_speed=args.easy_model_speed,
        easy_truth_speed=args.easy_truth_speed,
        easy_model_beta=args.easy_model_beta,
        easy_truth_beta=args.easy_truth_beta,
    )
    requested_variant_names = parse_variant_selection(args.eval_variants)
    variants = [v for v in variants if v.name in requested_variant_names]
    if not variants:
        raise RuntimeError(f"No variants selected by --eval-variants={args.eval_variants!r}")
    if args.speed_envelope is None:
        max_abs_u = max(abs(control_bounds.lo), abs(control_bounds.hi))
        args.speed_envelope = max(
            math.hypot(spec.v, abs(spec.beta_u) * max_abs_u)
            for variant in variants
            for spec in (variant.model, variant.truth)
        )
    shield_cfg = ShieldConfig(
        cbvf_activate_margin=args.cbvf_activate_margin,
        cp_activate_margin=args.cp_activate_margin,
    )

    script_dir = Path(__file__).resolve().parent
    cache_dir = (
        Path(args.cache_dir).expanduser().resolve()
        if args.cache_dir is not None
        else (script_dir / "cbvf_cache")
    )
    plot_dir = (
        Path(args.plot_dir).expanduser().resolve()
        if args.plot_dir is not None
        else (script_dir / "plots")
    )

    if args.target_shape != "signed_distance":
        print(
            "WARNING: target_shape is not signed_distance. The rollout unsafe flag "
            "uses signed distance, so paper experiments should use --target-shape signed_distance."
        )
    else:
        print("Safety function check: rollout l(x) and CBVF target both use signed distance to the obstacle boundary.")

    print(
        f"Shield activation margins: CBVF={args.cbvf_activate_margin:.3f}, "
        f"CP={args.cp_activate_margin:.3f}. CBVF default is least-restrictive; "
        "CP uses the active Mondrian buffer as the QP RHS."
    )
    print(
        f"Mondrian radial clearance edges: {mondrian_partition.clearance_edges}; "
        f"delta_traj={args.delta_traj:.4f}; epsilon_grid={args.epsilon_grid:.6f}; "
        f"epsilon_inter={args.epsilon_inter:.6f}; C_bar={args.speed_envelope:.6f}"
    )
    print(
        "CBVF table policy: "
        + ("rebuild from scratch" if args.recompute_cbvf else "use compatible cache when available")
    )
    print(f"Evaluation seeds: {seeds}")
    print(f"Selected deployment variants: {[v.name for v in variants]}")
    if len(variants) == 1:
        print(
            "Matched-policy protocol: this run evaluates the supplied RL checkpoint "
            f"only in {variants[0].name}. Run a separate command with the corresponding "
            "checkpoint for hard/easy mismatch experiments."
        )

    # ------------------------------------------------------------------
    # Build / load CBVF tables (one per unique model spec)
    # ------------------------------------------------------------------
    tables_by_model: Dict[Tuple[float, float], CBVFTable] = {}
    paths_by_model: Dict[Tuple[float, float], Path] = {}
    built_flags: Dict[Tuple[float, float], bool] = {}
    for variant in variants:
        key = (variant.model.v, variant.model.beta_u)
        if key not in tables_by_model:
            table, built_now, path = prepare_cbvf_table(
                cache_dir=cache_dir,
                world=world,
                control_bounds=control_bounds,
                model_spec=variant.model,
                gamma=args.gamma,
                dt=args.dt,
                horizon=args.horizon,
                cbvf_dt=args.cbvf_dt,
                nx=args.cbvf_nx,
                ny=args.cbvf_ny,
                nth=args.cbvf_nth,
                accuracy=args.solver_accuracy,
                recompute=args.recompute_cbvf,
                odp_root=args.odp_root,
                solver_file=args.solver_file,
                target_shape=args.target_shape,
                target_clip=args.target_clip,
                target_scale=args.target_scale,
                max_abs_table=args.max_abs_table,
                max_abs_grad=args.max_abs_grad,
                max_abs_solver_value=args.max_abs_solver_value,
                allow_coarse_fallback=args.allow_coarse_fallback,
                max_est_runtime_gb=args.max_est_runtime_gb,
                use_time_invariant_cbvf=args.time_invariant_cbvf,
            )
            tables_by_model[key] = table
            paths_by_model[key] = path
            built_flags[key] = built_now

    # ------------------------------------------------------------------
    # Build the rollout time axis for the finite-horizon experiment.
    # ------------------------------------------------------------------
    sim_dt = float(args.dt)
    effective_horizon = int(args.horizon)
    tau_eval = np.linspace(-effective_horizon * sim_dt, 0.0, effective_horizon + 1, dtype=float)
    effective_dt = float(tau_eval[1] - tau_eval[0])

    # ------------------------------------------------------------------
    # Banner
    # ------------------------------------------------------------------
    print("\n=== Mondrian conformal CBVF shielding experiment ===")
    print(
        f"γ={args.gamma}, delta_traj={args.delta_traj}, n_calib={args.n_calib}, "
        f"n_eval={args.n_eval}, seeds={seeds}, horizon={effective_horizon}, "
        f"rollout_dt={effective_dt:.4f}, cbvf_dt={args.cbvf_dt}"
    )
    print(
        "Calibration: regional rollout maxima under the baseline CBVF shield, "
        "followed by Mondrian split-conformal quantiles.\n"
        "Regional risk: one Bonferroni term per leaf of a mixed-depth strict-tree cut, "
        "with local promotion of sparse branches.\n"
        "Baseline CBVF: activation at B<=cbvf_activate_margin with rhs=0.\n"
        "CP shield: activation at B<=cp_activate_margin with rhs=max reachable "
        "regional buffer + epsilon_inter.\n"
        "The conformal buffer tightens the QP constraint but does not shift activation.\n"
        "CBVF is finite-horizon, so DtB is retained in the online constraint."
    )
    print()

    all_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    all_seed_results: Dict[str, List[Dict[str, object]]] = {}
    all_calibration_results: Dict[str, Dict[str, object]] = {}
    # stores first-seed eval rollout logs per variant per method for trajectory plots
    first_seed_eval_logs: Dict[str, Dict[str, list]] = {}

    # ------------------------------------------------------------------
    # Per-variant experiment
    # ------------------------------------------------------------------
    for variant in variants:
        cbvf = tables_by_model[(variant.model.v, variant.model.beta_u)]
        print(f"=== Variant: {variant.name} ===")
        print_table_diagnostics(cbvf, label=variant.name)
        print(
            f"  model=(v={variant.model.v:.2f}, β={variant.model.beta_u:.2f}), "
            f"truth=(v={variant.truth.v:.2f}, β={variant.truth.beta_u:.2f}), "
            f"cbvf_path={paths_by_model[(variant.model.v, variant.model.beta_u)]}"
        )

        policy = build_nominal_policy(args, world=world, control_bounds=control_bounds)
        if hasattr(policy, "describe"):
            print(f"  nominal controller: {policy.describe()}")

        variant_seed_results: List[Dict[str, Dict[str, float]]] = []
        variant_seed_records: List[Dict[str, object]] = []
        xi_values: List[float] = []
        calib_unsafe_values: List[float] = []

        for seed in seeds:
            random.seed(seed)
            np.random.seed(seed)
            print(f"  -- seed {seed} --")

            calib_states = make_shared_initial_states(
                world=world, control_bounds=control_bounds, variant=variant,
                n_states=args.n_calib, seed=1000 + seed,
                dt=effective_dt, horizon=effective_horizon,
                fixed_start=args.fixed_start,
                start_jitter_xy=args.start_jitter_xy,
                start_jitter_theta=args.start_jitter_theta,
                cbvf=cbvf,
                tau=tau_eval,
            )
            eval_states = make_shared_initial_states(
                world=world, control_bounds=control_bounds, variant=variant,
                n_states=args.n_eval, seed=2000 + seed,
                dt=effective_dt, horizon=effective_horizon,
                fixed_start=args.fixed_start,
                start_jitter_xy=args.start_jitter_xy,
                start_jitter_theta=args.start_jitter_theta,
                cbvf=cbvf,
                tau=tau_eval,
            )
            heldout_split_check = assert_disjoint_initial_state_arrays(
                calibration_states=calib_states,
                evaluation_states=eval_states,
            )

            if len(seeds) == 1 or seed == seeds[0]:
                print_start_diagnostics(
                    world=world,
                    cbvf=cbvf,
                    tau=tau_eval,
                    calib_states=calib_states,
                    eval_states=eval_states,
                )

            # ---- Offline Mondrian CP calibration ----
            mondrian_calibration, calib_stats = calibrate_mondrian(
                world=world,
                control_bounds=control_bounds,
                variant=variant,
                cbvf=cbvf,
                policy=policy,
                tau=tau_eval,
                init_states=calib_states,
                gamma=args.gamma,
                delta_traj=args.delta_traj,
                shield_cfg=shield_cfg,
                continue_after_unsafe=args.calib_continue_after_unsafe,
                partition=mondrian_partition,
                epsilon_grid=args.epsilon_grid,
            )
            xi_max = float(mondrian_calibration.max_buffer)
            xi_values.append(xi_max)
            calib_unsafe_values.append(float(calib_stats["mean_calib_unsafe"]))
            if len(seeds) == 1:
                print_calibration_stats(calib_stats, variant_name=variant.name)
            else:
                if calib_stats["nonzero_frac"] < 0.10 and variant.name == "AlignedModel":
                    degenerate_msg = "expected degenerate eta; CP≈CBVF"
                elif calib_stats["nonzero_frac"] < 0.10:
                    degenerate_msg = "WARNING: degenerate eta"
                else:
                    degenerate_msg = ""
                print(
                    f"    Mondrian CP: M_eff={mondrian_calibration.n_effective_regions}, "
                    f"xi_max={xi_max:.6f}, "
                    f"score_mean={calib_stats['score_mean']:.6f}, "
                    f"nonzero_frac={calib_stats['nonzero_frac']:.3f}, "
                    f"calib_unsafe_rate={calib_stats['mean_calib_unsafe']:.3f}"
                    + (f" ({degenerate_msg})" if degenerate_msg else "")
                )

            if not np.isfinite(xi_max) or xi_max > args.max_reasonable_xi:
                raise RuntimeError(
                    f"Maximum regional xi_hat={xi_max:.6g} is unreasonably large. "
                    "Check the CBVF table and gradient magnitudes."
                )

            # ---- Evaluation ----
            result_seed: Dict[str, Dict[str, float]] = {}
            for method in ["nominal", "cbvf", "cp"]:
                logs = evaluate_method(
                    world=world,
                    control_bounds=control_bounds,
                    variant=variant,
                    cbvf=cbvf,
                    policy=policy,
                    tau=tau_eval,
                    init_states=eval_states,
                    method=method,
                    xi=0.0,
                    gamma=args.gamma,
                    shield_cfg=shield_cfg,
                    mondrian_calibration=mondrian_calibration,
                    speed_envelope=args.speed_envelope,
                    epsilon_inter=args.epsilon_inter,
                )
                stats = summarize(logs)
                result_seed[method] = stats
                # store first-seed logs for trajectory plots
                if seed == seeds[0]:
                    first_seed_eval_logs.setdefault(variant.name, {})[method] = logs
                print(
                    f"    {method:8s} | "
                    f"unsafe={stats['unsafe_rate']:.3f}, "
                    f"goal={stats['goalconda _rate']:.3f}, "
                    f"return={stats['mean_return']:.2f}, "
                    f"interventions={stats['mean_interventions']:.1f}, "
                    f"min_l={stats['mean_min_l']:.4f}, "
                    f"buffer_exceed={stats['buffer_exceedance_rate']:.3f}, "
                    f"assumption5_episode_rate={stats['assumption5_violation_episode_rate']:.3f}"
                )

            if (
                int(result_seed["cbvf"]["n_buffer_scored_rollouts"]) != len(eval_states)
                or not np.isfinite(result_seed["cbvf"]["simultaneous_buffer_coverage"])
            ):
                raise RuntimeError(
                    "Held-out reference coverage was requested without attaching "
                    "the Mondrian calibration object for post-hoc scoring."
                )
            print(
                "    held-out uncalibrated-reference simultaneous coverage="
                f"{result_seed['cbvf']['simultaneous_buffer_coverage']:.3f}; "
                "calibrated-deployment buffer exceedance="
                f"{result_seed['cp']['buffer_exceedance_rate']:.3f}"
            )

            variant_seed_results.append(result_seed)
            variant_seed_records.append({
                "seed": int(seed),
                "xi_hat": float(xi_max),
                "heldout_split_check": heldout_split_check,
                "mondrian_calibration": mondrian_calibration.as_dict(),
                "calibration": calib_stats,
                "results": result_seed,
            })

        result_variant = aggregate_variant_seed_results(variant_seed_results)
        all_results[variant.name] = result_variant
        all_seed_results[variant.name] = variant_seed_records
        xi_arr = np.asarray(xi_values, dtype=float)
        calib_arr = np.asarray(calib_unsafe_values, dtype=float)
        all_calibration_results[variant.name] = {
            "buffer_summary": "maximum regional buffer per seed",
            "xi_hat_mean": float(np.mean(xi_arr)),
            "xi_hat_std_across_seeds": float(np.std(xi_arr, ddof=1)) if len(xi_arr) > 1 else 0.0,
            "calib_unsafe_rate_mean": float(np.mean(calib_arr)),
            "calib_unsafe_rate_std_across_seeds": float(np.std(calib_arr, ddof=1)) if len(calib_arr) > 1 else 0.0,
        }

        print("  Aggregate across seeds:")
        for method in ["nominal", "cbvf", "cp"]:
            stats = result_variant[method]
            print(
                f"    {method:8s} | "
                f"unsafe_rate={fmt_mean_std(stats, 'unsafe_rate')}, "
                f"goal_rate={fmt_mean_std(stats, 'goal_rate')}, "
                f"mean_return={fmt_mean_std(stats, 'mean_return', digits=2)}, "
                f"interventions={fmt_mean_std(stats, 'mean_interventions', digits=1)}, "
                f"min_l={fmt_mean_std(stats, 'mean_min_l')}"
            )
        print(
            f"    max regional xi_hat={all_calibration_results[variant.name]['xi_hat_mean']:.6f}±"
            f"{all_calibration_results[variant.name]['xi_hat_std_across_seeds']:.6f}"
        )
        print(
            "    held-out reference coverage="
            f"{fmt_mean_std(result_variant['cbvf'], 'simultaneous_buffer_coverage')}; "
            "calibrated deployment exceedance="
            f"{fmt_mean_std(result_variant['cp'], 'buffer_exceedance_rate')}"
        )
        print(
            "    Assumption-5 violation episode rate: "
            f"CBVF={fmt_mean_std(result_variant['cbvf'], 'assumption5_violation_episode_rate')}, "
            f"CP={fmt_mean_std(result_variant['cp'], 'assumption5_violation_episode_rate')}"
        )

        if result_variant["nominal"]["unsafe_rate"] == 0.0:
            print("    NOTE: nominal unsafe_rate is zero for this variant; the shield may have little to correct.")
        if result_variant["cbvf"]["unsafe_rate"] == 0.0 and variant.name == "HardMismatch":
            print(
                "    WARNING: HardMismatch still has CBVF unsafe_rate=0. "
                "To make the CP contribution visible, try increasing --hard-model-beta, "
                "decreasing --hard-truth-beta, or adding speed optimism with "
                "--hard-model-speed 0.70 --hard-truth-speed 0.50."
            )
        if result_variant["cbvf"]["mean_interventions"] == 0.0:
            print(
                "    WARNING: CBVF made zero interventions. Check the nominal mean_min_B "
                f"({result_variant['nominal']['mean_min_B']:.4f}) against cbvf_activate_margin "
                f"({args.cbvf_activate_margin:.4f}). If this is unexpected, check the checkpoint and CBVF table."
            )
        print()

    # ------------------------------------------------------------------
    # Save numeric results before plotting.
    # ------------------------------------------------------------------
    metadata = {
        "rl_checkpoint_dir": str(Path(args.rl_checkpoint_dir).expanduser().resolve()),
        "rl_checkpoint_step": str(args.rl_checkpoint_step),
        "target_shape": str(args.target_shape),
        "n_calib": int(args.n_calib),
        "n_eval": int(args.n_eval),
        "horizon": int(args.horizon),
        "dt": float(args.dt),
        "calibration_method": "mondrian_split_conformal",
        "delta_traj": float(args.delta_traj),
        "mondrian_partition": "radial_clearance",
        "sparse_region_fallback": "local_promotion_on_strict_binary_tree_cut",
        "heldout_split": "separate_draws_plus_exact_array_value_disjointness_check",
        "heldout_exchangeability_basis": "same fixed uncalibrated reference-policy data-generating protocol",
        "coverage_scoring": "raw_realized_regional_eta_against_epsilon_grid_inflated_calibration_buffers",
        "qp_infeasibility_logging": "original_requested_rhs_with_assumption5_violation_counter",
        "disturbance_model": "disturbance_free_dubins_with_explicit_guard",
        "tube_region_rule": "closed_interval_including_boundary_touching_neighbors",
        "rollout_time_rule": "exact_horizon_alignment_without_terminal_slice_clamping",
        "mondrian_clearance_edges": [
            float(v) for v in mondrian_partition.clearance_edges
        ],
        "epsilon_grid": float(args.epsilon_grid),
        "epsilon_inter": float(args.epsilon_inter),
        "speed_envelope": float(args.speed_envelope),
        "recompute_cbvf": bool(args.recompute_cbvf),
        "cbvf_activate_margin": float(args.cbvf_activate_margin),
        "cp_activate_margin": float(args.cp_activate_margin),
        "seed_argument": int(args.seed),
        "seeds": [int(s) for s in seeds],
        "hard_model_speed": float(args.hard_model_speed),
        "hard_truth_speed": float(args.hard_truth_speed),
        "hard_model_beta": float(args.hard_model_beta),
        "hard_truth_beta": float(args.hard_truth_beta),
        "easy_model_speed": float(args.easy_model_speed),
        "easy_truth_speed": float(args.easy_truth_speed),
        "easy_model_beta": float(args.easy_model_beta),
        "easy_truth_beta": float(args.easy_truth_beta),
        "eval_variants_argument": str(args.eval_variants),
        "evaluated_variants": [str(v.name) for v in variants],
        "matched_policy_protocol": True,
    }
    json_path = (
        Path(args.results_json).expanduser().resolve()
        if args.results_json is not None
        else (plot_dir / "experiment_results.json")
    )
    if not args.no_plots or args.results_json is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "metadata": metadata,
                "results_mean_std_across_seeds": all_results,
                "calibration_mean_std_across_seeds": all_calibration_results,
                "per_seed_results": all_seed_results,
            }, f, indent=2)
        print(f"Saved numeric results to {json_path}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    if not args.no_plots:
        plot_dir.mkdir(parents=True, exist_ok=True)
        print("Plot sanity check: goal_rate values by variant/method")
        for vname, vres in all_results.items():
            print(
                f"  {vname}: "
                f"nominal={vres['nominal']['goal_rate']:.3f}, "
                f"cbvf={vres['cbvf']['goal_rate']:.3f}, "
                f"cp={vres['cp']['goal_rate']:.3f}"
            )
        plot_safety_comparison(
            all_results,
            output_path=plot_dir / "safety_comparison.png",
            show=args.show_plots,
        )
        plot_grouped_bars(
            all_results,
            metric="goal_rate",
            ylabel="Goal-reaching rate",
            title="Goal-reaching rate",
            output_path=plot_dir / "goal_rate.png",
            show=args.show_plots,
            ylim=(0.0, 1.0),
        )
        plot_grouped_bars(
            all_results,
            metric="unsafe_rate",
            ylabel="Unsafe episode rate",
            title="Unsafe episode rate",
            output_path=plot_dir / "unsafe_rate.png",
            show=args.show_plots,
            ylim=(0.0, 1.0),
        )
        plot_grouped_bars(
            all_results,
            metric="mean_return",
            ylabel="Mean rollout return",
            title="Mean rollout return by mismatch variant",
            output_path=plot_dir / "mean_return.png",
            show=args.show_plots,
        )
        plot_grouped_bars(
            all_results,
            metric="mean_steps",
            ylabel="Mean rollout length (steps)",
            title="Episode length explains return differences",
            output_path=plot_dir / "episode_length.png",
            show=args.show_plots,
            ylim=(0.0, float(args.horizon)),
        )
        plot_grouped_bars(
            all_results,
            metric="mean_interventions",
            ylabel="Mean interventions per rollout",
            title="Shield interventions per rollout",
            output_path=plot_dir / "interventions.png",
            show=args.show_plots,
        )
        plot_grouped_bars(
            all_results,
            metric="total_unsafe_episodes",
            ylabel="Unsafe episodes",
            title="Unsafe episodes out of n_eval",
            output_path=plot_dir / "unsafe_episodes.png",
            show=args.show_plots,
            ylim=(0.0, float(args.n_eval)),
        )

        # --- Trajectory plots (one per variant, using first seed's eval logs) ---
        for vname, method_logs in first_seed_eval_logs.items():
            plot_trajectories_for_variant(
                variant_name=vname,
                eval_logs=method_logs,
                world=world,
                output_path=plot_dir / f"trajectories_{vname}.png",
                show=args.show_plots,
                n_trajs=10,
            )

        print(f"Saved plots to {plot_dir}")


if __name__ == "__main__":
    main()
