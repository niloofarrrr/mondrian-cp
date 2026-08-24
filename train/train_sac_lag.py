#! /usr/bin/env python
"""Train SAC/SAC-Lag behind the finalized Mondrian conformal CBVF shield.

The ordering in this entry point is deliberate and enforced:

1. create a fresh RL agent (no pretrained checkpoint),
2. rebuild the finite-horizon CBVF table from scratch,
3. calibrate a pre-training shield before collecting any RL transitions,
4. train with every proposal passed through that calibrated shield, then
5. freeze the learned policy, recalibrate on fresh baseline-CBVF rollouts from
   that exact policy, and evaluate on a third, disjoint set of seeds.

``conformal_shield.py`` is treated as finalized.  This file imports and reuses
its CBVF solver, strict-tree Mondrian partition/fallback, regional conformal
quantiles, reachable-region buffer rule, and exact two-dimensional CBVF-QP.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# When this file is executed as ``python train/train_sac_lag.py``, Python puts
# the ``train/`` directory on sys.path, but not always the project root.  The
# local packages used by this project (jaxrl5/, redexp/, and conformal_shield.py)
# live in the project root, so add it explicitly before importing them.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
import matplotlib
if "MPLBACKEND" not in os.environ:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from jaxrl5.agents import SACLearner, SACLagLearner
from jaxrl5.agents.agent import save_agent
from jaxrl5.data import ReplayBuffer
import redexp  # noqa: F401  registers ConformalDubins3d-v0


def _parse_hidden_dims(text: str) -> Tuple[int, ...]:
    return tuple(int(x.strip()) for x in text.split(",") if x.strip())


def _obs_from_state(state: np.ndarray, goal: Tuple[float, float]) -> np.ndarray:
    th = float(state[2])
    gx, gy = goal
    return np.array([state[0], state[1], np.cos(th), np.sin(th), gx, gy], dtype=np.float32)


def _reference_warmup_action(state: np.ndarray, beta_u: float) -> np.ndarray:
    """Pure-pursuit demonstrator used only to seed off-policy replay."""
    state = np.asarray(state, dtype=float)
    waypoint = np.array([-0.65, 0.65], dtype=float)
    goal = np.array([2.0, 2.0], dtype=float)
    # The y-threshold is a memoryless, monotone phase switch: unlike a radius
    # test it cannot switch back to the waypoint after the car departs it.
    target = waypoint if state[1] < 0.45 else goal
    desired = math.atan2(target[1] - state[1], target[0] - state[0])
    error = (desired - float(state[2]) + math.pi) % (2.0 * math.pi) - math.pi
    yaw = np.clip(1.5 * error / max(float(beta_u), 1e-6), -1.0, 1.0)
    return np.array([1.0, yaw], dtype=np.float32)


def _make_agent(args, env, hidden_dims: Sequence[int]):
    if args.algorithm == "sac":
        return SACLearner.create(
            args.seed,
            env.observation_space,
            env.action_space,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            temp_lr=args.temp_lr,
            hidden_dims=hidden_dims,
            discount=args.discount,
            tau=args.tau,
            init_temperature=args.init_temperature,
        )
    return SACLagLearner.create(
        args.seed,
        env.observation_space,
        env.action_space,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        temp_lr=args.temp_lr,
        lag_lr=args.lag_lr,
        hidden_dims=hidden_dims,
        discount=args.discount,
        tau=args.tau,
        init_temperature=args.init_temperature,
        init_lag=args.init_lag,
        cost_limit=args.cost_limit,
    )


class AgentPolicyCallable:
    """Fixed reference or frozen reference-plus-residual calibration policy.

    Flax agents are immutable.  Updating ``self.agent`` advances only this
    wrapper's RNG; it cannot update the training agent or its parameters.
    """

    def __init__(
        self,
        agent,
        goal: Tuple[float, float],
        lo: float,
        hi: float,
        model_beta_u: float,
        residual_scale: float = 0.0,
        use_frozen_residual: bool = False,
        stochastic: bool = True,
    ):
        self.agent = agent
        self.goal = goal
        self.lo = float(lo)
        self.hi = float(hi)
        self.model_beta_u = float(model_beta_u)
        self.residual_scale = float(residual_scale)
        self.use_frozen_residual = bool(use_frozen_residual)
        self.stochastic = bool(stochastic)

    def __call__(self, state: np.ndarray) -> np.ndarray:
        reference = _reference_warmup_action(state, self.model_beta_u)
        if not self.use_frozen_residual:
            return np.clip(reference, self.lo, self.hi)
        obs = _obs_from_state(np.asarray(state, dtype=float), self.goal)
        residual, self.agent = self.agent.eval_actions(obs)
        return np.clip(
            reference + self.residual_scale * np.asarray(residual, dtype=float),
            self.lo,
            self.hi,
        )


class TrainingShield:
    """Mandatory Mondrian conformal CBVF-QP used for all training rollouts."""

    def __init__(self, args, env, agent_for_calibration):
        if agent_for_calibration is None:
            raise ValueError("The initial RL agent is required before calibration.")
        self.args = args
        try:
            import conformal_shield as cs
        except Exception as exc:
            raise ImportError(
                "Shielded training requires the finalized conformal_shield.py in the project root."
            ) from exc

        self.cs = cs
        cfg = env.unwrapped.cfg
        self.goal = tuple(cfg.goal)
        self.initial_state = tuple(float(value) for value in cfg.initial_state)
        self.world = cs.World(
            xmin=float(cfg.xmin), xmax=float(cfg.xmax),
            ymin=float(cfg.ymin), ymax=float(cfg.ymax),
            goal=tuple(cfg.goal), goal_radius=float(cfg.goal_radius),
            obstacle_center=tuple(cfg.obstacle_center),
            obstacle_radius=float(cfg.obstacle_radius),
            robot_radius=float(cfg.robot_radius),
            init_x_range=(self.initial_state[0], self.initial_state[0]),
            init_y_range=(self.initial_state[1], self.initial_state[1]),
            init_theta_range=(self.initial_state[2], self.initial_state[2]),
        )
        world_initial_state = (
            self.world.init_x_range[0],
            self.world.init_y_range[0],
            self.world.init_theta_range[0],
        )
        if not np.allclose(world_initial_state, self.initial_state, rtol=0.0, atol=1e-12):
            raise RuntimeError(
                "Calibration World and Gym environment initial states are inconsistent."
            )
        self.control_bounds = cs.Interval(lo=float(cfg.u_min), hi=float(cfg.u_max))
        self.variant = cs.DynamicsVariant(
            name="TrainingDynamics",
            model=cs.DynamicsSpec(
                v=float(args.cbvf_model_speed),
                beta_u=float(args.cbvf_model_beta_u),
                v_min=float(args.cbvf_model_speed_min),
            ),
            truth=cs.DynamicsSpec(v=float(cfg.speed), beta_u=float(cfg.beta_u), v_min=float(cfg.speed_min)),
        )
        self.gamma = float(args.gamma)
        self.residual_reference_scale = float(args.residual_reference_scale)
        self.dt = float(args.dt)
        self.horizon = int(args.horizon)
        self.cbvf_terminal_guard = float(args.cbvf_terminal_guard)
        cbvf_lookback = self.horizon * self.dt + self.cbvf_terminal_guard
        self.tau = np.linspace(
            -cbvf_lookback,
            -self.cbvf_terminal_guard,
            self.horizon + 1,
            dtype=float,
        )
        self.shield_cfg = cs.ShieldConfig(
            cbvf_activate_margin=float(args.cbvf_activate_margin),
            cp_activate_margin=float(args.cp_activate_margin),
        )
        self.partition = cs.MondrianPartition(
            clearance_edges=cs.parse_strictly_increasing_floats(
                args.mondrian_clearance_edges,
                "--mondrian-clearance-edges",
            ),
            obstacle_center=self.world.obstacle_center,
            inflated_obstacle_radius=float(
                self.world.obstacle_radius + self.world.robot_radius
            ),
        )
        max_abs_u = max(abs(self.control_bounds.lo), abs(self.control_bounds.hi))
        self.speed_envelope = (
            float(args.speed_envelope)
            if args.speed_envelope is not None
            else max(
                math.hypot(spec.v, abs(spec.beta_u) * max_abs_u)
                for spec in (self.variant.model, self.variant.truth)
            )
        )
        self.epsilon_inter = float(args.epsilon_inter)
        self.calibration_stochastic_policy = bool(args.calibration_stochastic_policy)
        self.allow_infeasible_cp_diagnostic_run = bool(
            args.allow_infeasible_cp_diagnostic_run
        )
        self._is_shielding = False

        cache_dir = (
            Path(args.cache_dir).expanduser().resolve()
            if args.cache_dir
            else (PROJECT_ROOT / "cbvf_cache")
        )
        self.cbvf, self.cbvf_recomputed, self.cbvf_path = cs.prepare_cbvf_table(
            cache_dir=cache_dir,
            world=self.world,
            control_bounds=self.control_bounds,
            model_spec=self.variant.model,
            gamma=self.gamma,
            dt=self.dt,
            horizon=int(math.ceil(cbvf_lookback / self.dt)),
            cbvf_dt=float(args.cbvf_dt),
            nx=int(args.cbvf_nx),
            ny=int(args.cbvf_ny),
            nth=int(args.cbvf_nth),
            accuracy=str(args.solver_accuracy),
            recompute=True,
            odp_root=args.odp_root,
            solver_file=args.solver_file,
            target_shape=str(args.target_shape),
            target_clip=float(args.target_clip),
            target_scale=0.0,
            max_abs_table=float(args.max_abs_table),
            max_abs_grad=float(args.max_abs_grad),
            max_abs_solver_value=float(args.max_abs_solver_value),
            allow_coarse_fallback=bool(args.allow_coarse_fallback),
            max_est_runtime_gb=float(args.max_est_runtime_gb),
            use_time_invariant_cbvf=bool(args.time_invariant_cbvf),
        )
        if not self.cbvf_recomputed:
            raise RuntimeError(
                "Fresh CBVF recomputation was required, but prepare_cbvf_table did not rebuild it."
            )

        # Calibration is completed before the caller resets the RL environment
        # for episode one.  The initial actor is fixed inside policy_for_calib,
        # and every rollout uses the baseline CBVF shield with xi=0 inside the
        # finalized calibrate_mondrian implementation.
        policy_for_calib = AgentPolicyCallable(
            agent_for_calibration,
            self.goal,
            cfg.u_min,
            cfg.u_max,
            self.variant.model.beta_u,
            residual_scale=float(args.residual_reference_scale),
            use_frozen_residual=bool(
                getattr(args, "post_training_recalibration", False)
            ),
            stochastic=self.calibration_stochastic_policy,
        )
        calibration_seed_offset = (
            30_000 if getattr(args, "post_training_recalibration", False) else 20_000
        )
        calibration_rng = np.random.default_rng(
            int(args.seed + calibration_seed_offset)
        )
        calibration_states: List[np.ndarray] = []
        for calibration_index in range(int(args.n_calib)):
            state = env.unwrapped.sample_initial_state(calibration_rng)
            B_initial, _, _ = self.cbvf.value_grad_dt(state, float(self.tau[0]))
            if B_initial < 0.0:
                raise RuntimeError(
                    "The shared environment/calibration start distribution produced "
                    f"B(x0,t0)={B_initial:.6f}<0 at calibration sample "
                    f"{calibration_index}. Reduce the start jitter or change the "
                    "configured initial state; calibration and RL starts are not "
                    "silently conditioned differently."
                )
            calibration_states.append(np.asarray(state, dtype=float))
        self.mondrian_calibration, self.calibration_stats = cs.calibrate_mondrian(
            world=self.world,
            control_bounds=self.control_bounds,
            variant=self.variant,
            cbvf=self.cbvf,
            policy=policy_for_calib,
            tau=self.tau,
            init_states=calibration_states,
            gamma=self.gamma,
            delta_traj=float(args.delta_traj),
            shield_cfg=self.shield_cfg,
            continue_after_unsafe=bool(args.calib_continue_after_unsafe),
            partition=self.partition,
            epsilon_grid=float(args.epsilon_grid),
        )
        self.xi_hat = float(self.mondrian_calibration.max_buffer)
        if not np.isfinite(self.xi_hat) or self.xi_hat > float(args.max_reasonable_xi):
            raise RuntimeError(
                f"Maximum regional xi_hat={self.xi_hat:.6g} is invalid or exceeds "
                f"--max-reasonable-xi={args.max_reasonable_xi:.6g}."
            )
        base_visit_counts = [
            int(value) for value in self.calibration_stats["base_visit_counts"]
        ]
        per_base_leaf_delta = float(args.delta_traj) / float(self.partition.n_base_regions)
        per_base_leaf_minimum = int(
            cs._minimum_conformal_count(per_base_leaf_delta)
        )
        near_obstacle_group = next(
            group
            for group in self.mondrian_calibration.effective_groups
            if 0 in group
        )
        self.mondrian_activity = {
            "n_base_regions": int(self.partition.n_base_regions),
            "n_effective_regions": int(self.mondrian_calibration.n_effective_regions),
            "mondrian_active": bool(self.mondrian_calibration.n_effective_regions > 1),
            "base_visit_counts": base_visit_counts,
            "near_obstacle_base_region": 0,
            "near_obstacle_clearance_upper": float(
                self.partition.clearance_edges[0]
            ),
            "near_obstacle_score_count": int(base_visit_counts[0]),
            "per_base_leaf_minimum_without_merging": per_base_leaf_minimum,
            "near_obstacle_retained_as_singleton": bool(tuple(near_obstacle_group) == (0,)),
            "near_obstacle_effective_group": [int(value) for value in near_obstacle_group],
            "coverage_claim_for_training_rollouts": "none_policy_changes_after_calibration",
        }

        # Separate object for model dynamics and exact QP calculations.  The
        # Gymnasium environment remains the true plant.
        self.qp_env = cs.DubinsCBVFEnv(
            world=self.world,
            control_bounds=self.control_bounds,
            variant=self.variant,
            dt=self.dt,
            horizon=self.horizon,
        )
        self.offline_feasibility_audit = self._audit_cp_feasibility_grid()
        print(
            "Mandatory training shield ready: method=mondrian_cp, "
            f"model=(v={self.variant.model.v}, beta={self.variant.model.beta_u}), "
            f"truth=(v={self.variant.truth.v}, beta={self.variant.truth.beta_u}), "
            f"M_effective={self.mondrian_calibration.n_effective_regions}, "
            f"xi_max={self.xi_hat:.6f}, table={self.cbvf_path}"
        )
        print(
            "Mondrian activity check: "
            f"near_obstacle_scores={base_visit_counts[0]}/"
            f"{per_base_leaf_minimum} required for an unmerged base leaf, "
            f"near_group={tuple(near_obstacle_group)}, "
            f"M_effective={self.mondrian_calibration.n_effective_regions}"
        )
        print(
            "Offline CP-QP grid audit: "
            f"active_certified_nodes="
            f"{self.offline_feasibility_audit['n_active_certified_grid_nodes']}, "
            f"violations="
            f"{self.offline_feasibility_audit['n_infeasible_grid_nodes']}, "
            f"violation_rate="
            f"{self.offline_feasibility_audit['infeasible_grid_node_rate']:.6f}, "
            f"min_margin="
            f"{self.offline_feasibility_audit['min_feasibility_margin']:.6f}"
        )
        cs.print_calibration_stats(self.calibration_stats, variant_name=self.variant.name)

    def _audit_cp_feasibility_grid(self) -> Dict[str, Any]:
        """Discrete pre-training Assumption-5 gate on active certified table nodes.

        This checks every stored CBVF space-time grid node with B>=0 that can
        activate the CP filter, including its 0.05 hysteresis release band. It
        is an exhaustive audit of those discrete nodes, not a replacement for
        the paper's continuous verified-envelope/Lipschitz certificate.
        """
        cbvf = self.cbvf
        nx, ny, nth, nt = cbvf.B_table.shape
        regional_buffer_xy = np.empty((nx, ny), dtype=float)
        candidate_count_xy = np.empty((nx, ny), dtype=int)
        travel_radius = float(self.speed_envelope * self.dt)
        for ix, x_value in enumerate(cbvf.x_grid):
            for iy, y_value in enumerate(cbvf.y_grid):
                buffer_value, regions = self.mondrian_calibration.applied_buffer(
                    state=np.array([x_value, y_value, 0.0], dtype=float),
                    travel_radius=travel_radius,
                )
                regional_buffer_xy[ix, iy] = float(buffer_value)
                candidate_count_xy[ix, iy] = int(len(regions))

        theta = cbvf.th_grid[np.newaxis, np.newaxis, :]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        activation_upper = float(
            self.shield_cfg.activation_threshold("cp", self.xi_hat) + 0.05
        )
        tolerance = 1e-10
        n_active_certified = 0
        n_infeasible = 0
        n_uncalibrated_infeasible = 0
        min_margin = float("inf")
        min_uncalibrated_margin = float("inf")
        margin_sum = 0.0
        worst: Optional[Dict[str, Any]] = None

        # Audit only table slices that can be queried at a control decision.
        # The terminal guard deliberately keeps the deployed horizon away from
        # the terminal target slice, where steering has relative degree two and
        # an instantaneous first-order CBVF-QP need not be feasible.
        runtime_time_indices = sorted({
            int(np.argmin(np.abs(cbvf.tau - float(t))))
            for t in self.tau[:-1]
        })
        for time_index in runtime_time_indices:
            B_slice = cbvf.B_table[..., time_index]
            if nt == 1:
                DtB_slice = np.zeros_like(B_slice)
            elif time_index < nt - 1:
                DtB_slice = (
                    cbvf.B_table[..., time_index + 1] - B_slice
                ) / float(cbvf.dt)
            else:
                DtB_slice = (
                    B_slice - cbvf.B_table[..., time_index - 1]
                ) / float(cbvf.dt)
            heading_derivative = (
                cbvf.gx[..., time_index] * cos_theta
                + cbvf.gy[..., time_index] * sin_theta
            )
            sensitivity = np.abs(heading_derivative) + np.abs(
                cbvf.gth[..., time_index]
            )
            requested_rhs = (
                regional_buffer_xy[:, :, np.newaxis] * sensitivity
                + float(self.epsilon_inter)
            )
            model_speed_center = 0.5 * (
                self.variant.model.v + self.variant.model.v_min
            )
            model_speed_half = 0.5 * (
                self.variant.model.v - self.variant.model.v_min
            )
            base = (
                DtB_slice
                + heading_derivative * model_speed_center
                + float(self.gamma) * B_slice
            )
            a_throttle = heading_derivative * model_speed_half
            a_yaw = (
                cbvf.gth[..., time_index]
                * float(self.variant.model.beta_u)
            )
            max_u_psi = base + np.maximum(
                a_throttle * float(self.control_bounds.lo),
                a_throttle * float(self.control_bounds.hi),
            ) + np.maximum(
                a_yaw * float(self.control_bounds.lo),
                a_yaw * float(self.control_bounds.hi),
            )
            margins = max_u_psi - requested_rhs
            active_certified = (
                (B_slice >= 0.0) & (B_slice <= activation_upper)
            )
            active_count = int(np.count_nonzero(active_certified))
            if active_count == 0:
                continue
            active_margins = margins[active_certified]
            active_uncalibrated_margins = max_u_psi[active_certified]
            n_active_certified += active_count
            n_infeasible += int(np.count_nonzero(active_margins < -tolerance))
            n_uncalibrated_infeasible += int(
                np.count_nonzero(active_uncalibrated_margins < -tolerance)
            )
            min_uncalibrated_margin = min(
                min_uncalibrated_margin,
                float(np.min(active_uncalibrated_margins)),
            )
            margin_sum += float(np.sum(active_margins))
            slice_min = float(np.min(active_margins))
            if slice_min < min_margin:
                masked_margins = np.where(
                    active_certified,
                    margins,
                    np.inf,
                )
                flat_index = int(np.argmin(masked_margins))
                ix, iy, ith = np.unravel_index(
                    flat_index,
                    masked_margins.shape,
                )
                min_margin = slice_min
                worst = {
                    "state": [
                        float(cbvf.x_grid[ix]),
                        float(cbvf.y_grid[iy]),
                        float(cbvf.th_grid[ith]),
                    ],
                    "table_time": float(cbvf.tau[time_index]),
                    "B": float(B_slice[ix, iy, ith]),
                    "xi_hat_app": float(
                        requested_rhs[ix, iy, ith] - self.epsilon_inter
                    ),
                    "epsilon_inter": float(self.epsilon_inter),
                    "requested_xi": float(requested_rhs[ix, iy, ith]),
                    "max_u_psi": float(max_u_psi[ix, iy, ith]),
                    "feasibility_margin": float(margins[ix, iy, ith]),
                    "candidate_effective_region_count": int(
                        candidate_count_xy[ix, iy]
                    ),
                }

        if n_active_certified == 0:
            raise RuntimeError(
                "The offline CP-QP feasibility audit found no certified active "
                "CBVF grid nodes; the activation/envelope configuration is invalid."
            )
        return {
            "scope": (
                "all_stored_space_time_cbvf_grid_nodes_with_"
                "0_le_B_le_cp_activation_plus_hysteresis"
            ),
            "continuous_envelope_proof": False,
            "continuous_envelope_note": (
                "Discrete exhaustive gate plus runtime gating; a continuous "
                "Assumption-5 claim still requires the paper's validated "
                "interpolant/envelope bounds."
            ),
            "grid_shape": [int(nx), int(ny), int(nth), int(nt)],
            "runtime_time_indices_audited": [int(i) for i in runtime_time_indices],
            "cbvf_terminal_guard": float(self.cbvf_terminal_guard),
            "activation_upper": activation_upper,
            "n_active_certified_grid_nodes": int(n_active_certified),
            "n_infeasible_grid_nodes": int(n_infeasible),
            "n_uncalibrated_infeasible_grid_nodes": int(n_uncalibrated_infeasible),
            "min_uncalibrated_feasibility_margin": float(min_uncalibrated_margin),
            "infeasible_grid_node_rate": float(
                n_infeasible / n_active_certified
            ),
            "min_feasibility_margin": float(min_margin),
            "mean_feasibility_margin": float(
                margin_sum / n_active_certified
            ),
            "discrete_grid_gate_passed": bool(n_infeasible == 0),
            "worst_active_certified_grid_node": worst,
        }

    def reset_episode(self) -> None:
        self._is_shielding = False

    def metadata(self) -> Dict[str, Any]:
        return {
            "method": "mondrian_split_conformal_cbvf_qp",
            "calibration_controller": (
                "post_training_frozen_reference_plus_residual_policy"
                if getattr(self.args, "post_training_recalibration", False)
                else "fixed_nominal_model_reference_controller"
            ),
            "baseline_calibration_shield": "cbvf_qp_xi_0",
            "initial_state_distribution": {
                "base_state": [float(value) for value in self.initial_state],
                "sampler": "redexp.env.sample_initial_state_shared_by_calibration_and_rl",
                "requires_cbvf_certified_realizations": True,
            },
            "cbvf_recomputed": bool(self.cbvf_recomputed),
            "cbvf_path": str(self.cbvf_path),
            "model": {
                "speed": float(self.variant.model.v),
                "speed_min": float(self.variant.model.v_min),
                "beta_u": float(self.variant.model.beta_u),
            },
            "truth": {
                "speed": float(self.variant.truth.v),
                "speed_min": float(self.variant.truth.v_min),
                "beta_u": float(self.variant.truth.beta_u),
            },
            "speed_envelope": float(self.speed_envelope),
            "epsilon_inter": float(self.epsilon_inter),
            "cbvf_terminal_guard": float(self.cbvf_terminal_guard),
            "infeasible_cp_policy": (
                "explicit_diagnostic_least_violation_fallback"
                if self.allow_infeasible_cp_diagnostic_run
                else "abort_before_executing_infeasible_cp_action"
            ),
            "shield_config": self.shield_cfg.describe(self.xi_hat),
            "mondrian_calibration": self.mondrian_calibration.as_dict(),
            "calibration_stats": self.calibration_stats,
            "mondrian_activity": self.mondrian_activity,
            "offline_cp_qp_feasibility_audit": self.offline_feasibility_audit,
        }

    def project(
        self,
        state: np.ndarray,
        step_in_episode: int,
        u_nom: np.ndarray,
        method: str,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if method not in {"nominal", "cbvf", "cp"}:
            raise ValueError(f"Unsupported shield method: {method}")
        u_nom = np.clip(np.asarray(u_nom, dtype=float).reshape(2), self.control_bounds.lo, self.control_bounds.hi)
        if method == "nominal":
            return u_nom, {
                "active": 0.0,
                "intervention": 0.0,
                "infeasible": 0.0,
                "assumption5_violation": 0.0,
                "B": float("nan"),
                "regional_buffer": 0.0,
                "qp_rhs": 0.0,
                "xi_hat_app": 0.0,
                "epsilon_inter": 0.0,
                "requested_xi": 0.0,
                "max_u_psi": None,
                "feasibility_margin": None,
                "achieved_lhs": None,
                "achieved_constraint_margin": None,
                "a_u": None,
                "base_without_control": None,
                "fallback_used": False,
                "fallback_kind": "none",
                "fallback_is_actuator_rail": False,
                "fallback_bound": "none",
                "minimal_projection_verified": None,
                "candidate_effective_regions": (),
            }

        if not 0 <= int(step_in_episode) < self.horizon:
            raise RuntimeError(
                f"Episode step {step_in_episode} is outside the CBVF horizon {self.horizon}."
            )
        t_now = float(self.tau[int(step_in_episode)])
        state = np.asarray(state, dtype=float)
        B, _, _ = self.cbvf.value_grad_dt(state, t_now)

        regional_buffer = 0.0
        candidate_regions: Tuple[int, ...] = ()
        if method == "cp":
            regional_coefficient, candidate_regions = self.mondrian_calibration.applied_buffer(
                state=state,
                travel_radius=self.speed_envelope * self.dt,
            )
            regional_buffer = float(regional_coefficient) * float(
                self.cs.mismatch_sensitivity(state, t_now, self.cbvf)
            )
        qp_rhs = self.shield_cfg.qp_rhs(method, regional_buffer)
        if method == "cp":
            qp_rhs += self.epsilon_inter

        activation_threshold = self.shield_cfg.activation_threshold(
            method,
            regional_buffer,
        )
        release_buffer = 0.05 if self._is_shielding else 0.0
        active = bool(B <= activation_threshold + release_buffer)
        self._is_shielding = active
        if not active:
            return u_nom, {
                "active": 0.0,
                "intervention": 0.0,
                "infeasible": 0.0,
                "assumption5_violation": 0.0,
                "B": float(B),
                "regional_buffer": float(regional_buffer),
                "qp_rhs": float(qp_rhs),
                "xi_hat_app": float(regional_buffer),
                "epsilon_inter": float(self.epsilon_inter if method == "cp" else 0.0),
                "requested_xi": float(qp_rhs),
                "max_u_psi": None,
                "feasibility_margin": None,
                "achieved_lhs": None,
                "achieved_constraint_margin": None,
                "a_u": None,
                "base_without_control": None,
                "fallback_used": False,
                "fallback_kind": "none",
                "fallback_is_actuator_rail": False,
                "fallback_bound": "none",
                "minimal_projection_verified": None,
                "candidate_effective_regions": candidate_regions,
            }

        solution = self.cs.solve_cbvf_projection_exact(
            s=state,
            t=t_now,
            u_nom=u_nom,
            cbvf=self.cbvf,
            env=self.qp_env,
            gamma=self.gamma,
            xi=float(qp_rhs),
        )
        u_filtered = np.asarray(solution.u, dtype=float)
        feasibility_margin = float(
            solution.max_achievable_lhs - solution.requested_xi
        )
        fallback_used = bool(not solution.feasible)
        no_control_authority = bool(np.linalg.norm(solution.a_u) < 1e-10)
        at_lower_bound = bool(np.any(np.isclose(u_filtered, self.control_bounds.lo, atol=1e-10)))
        at_upper_bound = bool(np.any(np.isclose(u_filtered, self.control_bounds.hi, atol=1e-10)))
        fallback_is_actuator_rail = bool(
            fallback_used and not no_control_authority
            and (at_lower_bound or at_upper_bound)
        )
        if not fallback_used:
            fallback_kind = "none"
        elif no_control_authority:
            fallback_kind = "no_control_authority_keep_nominal"
        else:
            fallback_kind = "least_violation_affine_maximizer"
        fallback_bound = (
            "lower" if fallback_is_actuator_rail and at_lower_bound
            else "upper" if fallback_is_actuator_rail and at_upper_bound
            else "none"
        )

        minimal_projection_verified: Optional[bool] = None
        if solution.feasible:
            minimal_projection_verified = bool(
                solution.margin >= -1e-8
                and np.all(u_filtered >= self.control_bounds.lo - 1e-10)
                and np.all(u_filtered <= self.control_bounds.hi + 1e-10)
            )

        diagnostics = {
            "active": 1.0,
            "intervention": float(np.linalg.norm(u_filtered - u_nom) > 1e-8),
            "infeasible": float(fallback_used),
            "assumption5_violation": float(solution.assumption5_violation),
            "B": float(solution.B),
            "regional_buffer": float(regional_buffer),
            "qp_rhs": float(qp_rhs),
            "xi_hat_app": float(regional_buffer),
            "epsilon_inter": float(self.epsilon_inter if method == "cp" else 0.0),
            "requested_xi": float(solution.requested_xi),
            "max_u_psi": float(solution.max_achievable_lhs),
            "feasibility_margin": feasibility_margin,
            "achieved_lhs": float(solution.achieved_lhs),
            "achieved_constraint_margin": float(solution.margin),
            "a_u": [float(value) for value in solution.a_u],
            "base_without_control": float(solution.base_without_control),
            "qp_margin": float(solution.margin),
            "fallback_used": fallback_used,
            "fallback_kind": fallback_kind,
            "fallback_is_actuator_rail": fallback_is_actuator_rail,
            "fallback_bound": fallback_bound,
            "minimal_projection_verified": minimal_projection_verified,
            "candidate_effective_regions": candidate_regions,
        }
        if minimal_projection_verified is False:
            raise RuntimeError(
                "The finalized two-dimensional solver returned a feasible action "
                "that does not equal the analytic minimum-distance projection."
            )
        if (
            method == "cp"
            and fallback_used
            and not self.allow_infeasible_cp_diagnostic_run
        ):
            failure = {
                "state": [float(value) for value in state],
                "time": float(t_now),
                "episode_step": int(step_in_episode),
                "nominal_action": [float(value) for value in u_nom],
                "candidate_action": [float(value) for value in u_filtered],
                **{
                    key: diagnostics[key]
                    for key in (
                        "xi_hat_app",
                        "epsilon_inter",
                        "requested_xi",
                        "max_u_psi",
                        "feasibility_margin",
                        "fallback_kind",
                        "fallback_is_actuator_rail",
                        "fallback_bound",
                    )
                },
            }
            print("CP-QP infeasibility: " + json.dumps(failure, sort_keys=True))
            raise RuntimeError(
                "The calibrated CP-QP is infeasible at an active training/evaluation "
                "state, so the strict run was stopped before executing the "
                "least-violation fallback. This violates the claimed verified "
                "Assumption-5 envelope. Use --allow-infeasible-cp-diagnostic-run "
                "only for a labeled short diagnostic run."
            )
        return u_filtered, diagnostics


def _assert_time_limit_matches_horizon(env, horizon: int) -> None:
    wrapper_limit = getattr(env, "_max_episode_steps", None)
    spec_limit = getattr(getattr(env, "spec", None), "max_episode_steps", None)
    effective_limit = wrapper_limit if wrapper_limit is not None else spec_limit
    if effective_limit is None or int(effective_limit) != int(horizon):
        raise RuntimeError(
            "Gym TimeLimit and CBVF horizon are inconsistent: "
            f"TimeLimit={effective_limit}, CBVF horizon={horizon}."
        )
    if int(env.unwrapped.cfg.horizon) != int(horizon):
        raise RuntimeError(
            "Environment-internal horizon and CBVF horizon are inconsistent: "
            f"env={env.unwrapped.cfg.horizon}, CBVF={horizon}."
        )


def _reset_certified_env(
    env,
    shield: TrainingShield,
    seed: Optional[int] = None,
):
    obs, info = env.reset(seed=seed)
    state = np.asarray(env.unwrapped.state, dtype=float)
    B_initial, _, _ = shield.cbvf.value_grad_dt(state, float(shield.tau[0]))
    if B_initial < 0.0:
        raise RuntimeError(
            "The shared initial-state distribution produced an RL/evaluation "
            f"start with B(x0,t0)={B_initial:.6f}<0. Calibration and deployment "
            "are not silently conditioned on different start sets."
        )
    shield.reset_episode()
    info = dict(info)
    info["initial_cbvf_value"] = float(B_initial)
    return obs, info


def evaluate(
    agent,
    env,
    episodes: int,
    seed: int,
    shield: TrainingShield,
    method: str = "cp",
    evaluation_step: int = 0,
    qp_diagnostic_episodes: int = 0,
    qp_diagnostic_print_limit: int = 0,
    qp_diagnostics_path: Optional[Path] = None,
    capture_trajectory: bool = False,
) -> Dict[str, float]:
    returns, costs, goals, unsafes, lengths = [], [], [], [], []
    interventions, active_steps, infeasible_steps = [], [], []
    rail_fallback_steps, no_authority_fallback_steps = [], []
    candidate_region_counts: List[int] = []
    candidate_regions_seen: set = set()
    active_feasibility_margins: List[float] = []
    active_xi_hat_app: List[float] = []
    active_max_u_psi: List[float] = []
    minimal_projection_failures = 0
    rollout_coverages: List[float] = []
    safety_margins: List[float] = []
    total_safety_violations = 0
    diagnostic_records: List[Dict[str, Any]] = []
    printed_diagnostics = 0
    representative_trajectory: Dict[str, Any] = {
        "states": [], "nominal_actions": [], "applied_actions": [],
        "active": [], "feasibility_margin": [], "safe_margin": [],
    }
    for ep in range(episodes):
        obs, _ = _reset_certified_env(env, shield, seed=seed + ep)
        done = False
        ep_return = 0.0
        ep_cost = 0.0
        ep_goal = False
        ep_unsafe = False
        ep_len = 0
        ep_interventions = 0.0
        ep_active = 0.0
        ep_infeasible = 0.0
        ep_rail_fallback = 0.0
        ep_no_authority_fallback = 0.0
        ep_covered = True
        while not done:
            state_before = env.unwrapped.state.copy()
            action, agent = agent.eval_actions(obs)
            residual = np.asarray(action, dtype=np.float32).reshape(2)
            reference = _reference_warmup_action(
                state_before, shield.variant.model.beta_u
            )
            u_nom = np.clip(
                reference + shield.residual_reference_scale * residual,
                -1.0,
                1.0,
            ).astype(np.float32)
            if method != "nominal":
                u_applied, diag = shield.project(state_before, ep_len, u_nom, method=method)
                ep_interventions += float(diag["intervention"])
                ep_active += float(diag["active"])
                ep_infeasible += float(diag["infeasible"])
                regions: Tuple[int, ...] = ()
                if method == "cp":
                    regions = tuple(
                        int(value) for value in diag["candidate_effective_regions"]
                    )
                    candidate_region_counts.append(len(regions))
                    candidate_regions_seen.update(regions)
                    eta_now = shield.cs.eta_value(
                        state_before,
                        np.asarray(u_applied, dtype=float),
                        float(shield.tau[ep_len]),
                        shield.cbvf,
                        shield.qp_env,
                    )
                    ep_covered = bool(
                        ep_covered
                        and eta_now <= float(diag["xi_hat_app"]) + 1e-10
                    )
                if bool(diag["active"]):
                    active_feasibility_margins.append(
                        float(diag["feasibility_margin"])
                    )
                    active_xi_hat_app.append(float(diag["xi_hat_app"]))
                    active_max_u_psi.append(float(diag["max_u_psi"]))
                    ep_rail_fallback += float(
                        bool(diag["fallback_is_actuator_rail"])
                    )
                    ep_no_authority_fallback += float(
                        diag["fallback_kind"]
                        == "no_control_authority_keep_nominal"
                    )
                    minimal_projection_failures += int(
                        diag["minimal_projection_verified"] is False
                    )
                    if (
                        method == "cp"
                        and ep < int(qp_diagnostic_episodes)
                    ):
                        diagnostic_record = {
                            "evaluation_step": int(evaluation_step),
                            "episode": int(ep),
                            "episode_step": int(ep_len),
                            "state": [float(value) for value in state_before],
                            "nominal_action": [float(value) for value in u_nom],
                            "filtered_action": [float(value) for value in u_applied],
                            "feasible": bool(not diag["infeasible"]),
                            "xi_hat_app": float(diag["xi_hat_app"]),
                            "epsilon_inter": float(diag["epsilon_inter"]),
                            "requested_xi": float(diag["requested_xi"]),
                            "max_u_psi": float(diag["max_u_psi"]),
                            "feasibility_margin": float(
                                diag["feasibility_margin"]
                            ),
                            "achieved_lhs": float(diag["achieved_lhs"]),
                            "achieved_constraint_margin": float(
                                diag["achieved_constraint_margin"]
                            ),
                            "a_u": [float(value) for value in diag["a_u"]],
                            "base_without_control": float(
                                diag["base_without_control"]
                            ),
                            "fallback_used": bool(diag["fallback_used"]),
                            "fallback_kind": str(diag["fallback_kind"]),
                            "fallback_is_actuator_rail": bool(
                                diag["fallback_is_actuator_rail"]
                            ),
                            "fallback_bound": str(diag["fallback_bound"]),
                            "minimal_projection_verified": diag[
                                "minimal_projection_verified"
                            ],
                            "candidate_effective_regions": list(regions),
                        }
                        diagnostic_records.append(diagnostic_record)
                        if printed_diagnostics < int(
                            qp_diagnostic_print_limit
                        ):
                            print(
                                "CP-QP eval active-step: "
                                + json.dumps(diagnostic_record, sort_keys=True)
                            )
                            printed_diagnostics += 1
            else:
                u_applied = u_nom
                diag = {"active": 0.0, "feasibility_margin": None}
            if capture_trajectory and ep == 0:
                representative_trajectory["states"].append(
                    [float(value) for value in state_before]
                )
                representative_trajectory["nominal_actions"].append(
                    [float(value) for value in u_nom]
                )
                representative_trajectory["applied_actions"].append(
                    [float(value) for value in u_applied]
                )
                representative_trajectory["active"].append(float(diag["active"]))
                representative_trajectory["feasibility_margin"].append(
                    None if diag["feasibility_margin"] is None
                    else float(diag["feasibility_margin"])
                )
                representative_trajectory["safe_margin"].append(
                    float(env.unwrapped.safety_margin_value(state_before))
                )
            obs, reward, terminated, truncated, info = env.step(np.asarray(u_applied, dtype=np.float32))
            if method == "cp":
                next_state = env.unwrapped.state.copy()
                eta_next = shield.cs.eta_value(
                    next_state,
                    np.asarray(u_applied, dtype=float),
                    float(shield.tau[ep_len + 1]),
                    shield.cbvf,
                    shield.qp_env,
                )
                next_rho = shield.cs.mismatch_sensitivity(
                    next_state, float(shield.tau[ep_len + 1]), shield.cbvf
                )
                q_next, _ = shield.mondrian_calibration.applied_buffer(
                    state=next_state,
                    travel_radius=shield.speed_envelope * shield.dt,
                )
                ep_covered = bool(
                    ep_covered
                    and eta_next <= float(q_next) * float(next_rho) + 1e-10
                )
            done = bool(terminated or truncated)
            ep_return += float(reward)
            ep_cost += float(info.get("cost", 0.0))
            ep_goal = ep_goal or bool(info.get("reach_goal", False))
            ep_unsafe = ep_unsafe or bool(info.get("unsafe", False))
            safety_margins.append(float(info.get("safe_margin", float("nan"))))
            total_safety_violations += int(bool(info.get("unsafe", False)))
            ep_len += 1
        returns.append(ep_return)
        costs.append(ep_cost)
        goals.append(float(ep_goal))
        unsafes.append(float(ep_unsafe))
        lengths.append(ep_len)
        interventions.append(ep_interventions)
        active_steps.append(ep_active)
        infeasible_steps.append(ep_infeasible)
        rail_fallback_steps.append(ep_rail_fallback)
        no_authority_fallback_steps.append(ep_no_authority_fallback)
        if method == "cp":
            rollout_coverages.append(float(ep_covered))
    if diagnostic_records and qp_diagnostics_path is not None:
        with qp_diagnostics_path.open("a", encoding="utf-8") as diagnostic_file:
            for record in diagnostic_records:
                diagnostic_file.write(json.dumps(record, sort_keys=True) + "\n")
    total_active = float(np.sum(active_steps))
    total_infeasible = float(np.sum(infeasible_steps))
    total_rail_fallback = float(np.sum(rail_fallback_steps))
    result = {
        "return": float(np.mean(returns)),
        "cost": float(np.mean(costs)),
        "goal_rate": float(np.mean(goals)),
        "unsafe_rate": float(np.mean(unsafes)),
        "length": float(np.mean(lengths)),
        "interventions": float(np.mean(interventions)),
        "intervention_rate": float(
            np.sum(interventions) / max(1.0, float(np.sum(lengths)))
        ),
        "total_safety_violations": float(total_safety_violations),
        "minimum_safety_margin": float(np.nanmin(safety_margins)),
        "active_filter_steps": float(np.mean(active_steps)),
        "infeasible_steps": float(np.mean(infeasible_steps)),
        "active_filter_total": total_active,
        "infeasible_total": total_infeasible,
        "infeasible_rate_among_active": float(
            total_infeasible / max(1.0, total_active)
        ),
        "rail_fallback_steps": float(np.mean(rail_fallback_steps)),
        "rail_fallback_total": total_rail_fallback,
        "rail_fallback_rate_among_infeasible": float(
            total_rail_fallback / max(1.0, total_infeasible)
        ),
        "no_control_authority_fallback_steps": float(
            np.mean(no_authority_fallback_steps)
        ),
        "minimal_projection_failure_total": float(minimal_projection_failures),
        "min_active_feasibility_margin": (
            float(np.min(active_feasibility_margins))
            if active_feasibility_margins else 0.0
        ),
        "mean_active_feasibility_margin": (
            float(np.mean(active_feasibility_margins))
            if active_feasibility_margins else 0.0
        ),
        "mean_active_xi_hat_app": (
            float(np.mean(active_xi_hat_app)) if active_xi_hat_app else 0.0
        ),
        "epsilon_inter": float(shield.epsilon_inter if method == "cp" else 0.0),
        "mean_active_max_u_psi": (
            float(np.mean(active_max_u_psi)) if active_max_u_psi else 0.0
        ),
        "mean_candidate_effective_regions": (
            float(np.mean(candidate_region_counts))
            if candidate_region_counts else 0.0
        ),
        "max_candidate_effective_regions": (
            float(np.max(candidate_region_counts))
            if candidate_region_counts else 0.0
        ),
        "n_unique_candidate_effective_regions": float(len(candidate_regions_seen)),
        "empirical_simultaneous_coverage": (
            float(np.mean(rollout_coverages))
            if rollout_coverages else float("nan")
        ),
        "target_simultaneous_coverage": (
            float(1.0 - shield.mondrian_calibration.delta_traj)
            if method == "cp" else float("nan")
        ),
    }
    if capture_trajectory:
        result["representative_trajectory"] = representative_trajectory
    return result


def evaluate_all(
    agent,
    eval_env,
    episodes: int,
    seed: int,
    shield: TrainingShield,
    evaluation_step: int,
    qp_diagnostic_episodes: int,
    qp_diagnostic_print_limit: int,
    qp_diagnostics_path: Path,
    capture_trajectory: bool = False,
) -> Dict[str, Dict[str, float]]:
    training_hysteresis_state = shield._is_shielding
    try:
        return {
            method: evaluate(
                agent,
                eval_env,
                episodes=episodes,
                seed=seed,
                shield=shield,
                method=method,
                evaluation_step=evaluation_step,
                qp_diagnostic_episodes=(
                    qp_diagnostic_episodes if method == "cp" else 0
                ),
                qp_diagnostic_print_limit=(
                    qp_diagnostic_print_limit if method == "cp" else 0
                ),
                qp_diagnostics_path=(
                    qp_diagnostics_path if method == "cp" else None
                ),
                capture_trajectory=capture_trajectory,
            )
            for method in ("nominal", "cbvf", "cp")
        }
    finally:
        shield._is_shielding = training_hysteresis_state


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _plot_training_outputs(
    eval_records: Sequence[Dict[str, Any]],
    action_samples: Sequence[Dict[str, Any]],
    plot_dir: Path,
) -> List[str]:
    plot_dir.mkdir(parents=True, exist_ok=True)
    generated: List[str] = []

    if eval_records:
        steps = np.asarray([record["step"] for record in eval_records], dtype=float)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        method_labels = {
            "nominal": "unshielded actor diagnostic",
            "cbvf": "baseline CBVF-QP",
            "cp": "Mondrian CP-QP",
        }
        for method, color in (("nominal", "tab:gray"), ("cbvf", "tab:blue"), ("cp", "tab:green")):
            unsafe = [record["methods"][method]["unsafe_rate"] for record in eval_records]
            goal = [record["methods"][method]["goal_rate"] for record in eval_records]
            axes[0].plot(
                steps, unsafe, marker="o", label=method_labels[method], color=color
            )
            axes[1].plot(
                steps, goal, marker="o", label=method_labels[method], color=color
            )
        axes[0].set_title("Evaluation unsafe rate")
        axes[1].set_title("Evaluation goal rate")
        for axis in axes:
            axis.set_xlabel("training step")
            axis.set_ylim(-0.02, 1.02)
            axis.grid(alpha=0.25)
            axis.legend()
        axes[0].set_ylabel("rate")
        fig.tight_layout()
        eval_plot = plot_dir / "shielded_training_evaluation.png"
        fig.savefig(eval_plot, dpi=200, bbox_inches="tight")
        plt.close(fig)
        generated.append(str(eval_plot))

    if action_samples:
        sample_steps = np.asarray([item["step"] for item in action_samples], dtype=float)
        nominal = np.asarray(
            [item["nominal_action"] for item in action_samples], dtype=float
        )
        filtered = np.asarray(
            [item["filtered_action"] for item in action_samples], dtype=float
        )
        fallback_mask = np.asarray(
            [item["fallback_used"] for item in action_samples], dtype=bool
        )
        fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        axes[0].plot(
            sample_steps, nominal, linewidth=0.9,
            label="actor proposal", color="tab:gray"
        )
        axes[0].plot(
            sample_steps, filtered, linewidth=0.9,
            label="CP-QP executed", color="tab:green"
        )
        if np.any(fallback_mask):
            axes[0].scatter(
                sample_steps[fallback_mask],
                filtered[fallback_mask],
                marker="x",
                s=20,
                label="infeasible least-violation fallback",
                color="tab:red",
                zorder=3,
            )
        axes[0].set_ylabel("action")
        axes[0].legend()
        axes[0].grid(alpha=0.25)
        axes[1].plot(sample_steps, np.abs(filtered - nominal), linewidth=0.9, color="tab:red")
        if np.any(fallback_mask):
            axes[1].scatter(
                sample_steps[fallback_mask],
                np.abs(filtered - nominal)[fallback_mask],
                marker="x",
                s=20,
                color="black",
                zorder=3,
            )
        axes[1].set_xlabel("training step")
        axes[1].set_ylabel("|filtered - nominal|")
        axes[1].grid(alpha=0.25)
        fig.tight_layout()
        action_plot = plot_dir / "nominal_and_filtered_actions.png"
        fig.savefig(action_plot, dpi=200, bbox_inches="tight")
        plt.close(fig)
        generated.append(str(action_plot))

    return generated


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train SAC/SAC-Lag from scratch with mandatory pre-training "
            "Mondrian calibration and a conformal CBVF-QP on every rollout step."
        )
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=250_000)
    parser.add_argument(
        "--start-training",
        type=int,
        default=25_000,
        help=(
            "Replay warmup before the first optimizer update. Warmup actions "
            "always pass through the run's selected shield."
        ),
    )
    parser.add_argument(
        "--warmup-policy",
        choices=["actor", "reference"],
        default="reference",
        help=(
            "Proposal source before --start-training. Reference warmup only "
            "seeds replay; final evaluation is actor-only."
        ),
    )
    parser.add_argument(
        "--reference-proposal-steps",
        type=int,
        default=75_000,
        help=(
            "Number of initial collection steps using the reference proposal "
            "when --warmup-policy=reference; optimizer updates still begin at "
            "--start-training."
        ),
    )
    parser.add_argument(
        "--residual-reference-scale",
        type=float,
        default=0.10,
        help=(
            "Scale of the learned residual added to the nominal-model reference "
            "controller after guided collection; zero disables residual RL."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--utd-ratio", type=int, default=1)
    parser.add_argument("--eval-interval", type=int, default=10_000)
    parser.add_argument("--log-interval", type=int, default=1_000)
    parser.add_argument("--save-interval", type=int, default=25_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--periodic-eval-episodes", type=int, default=5)
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="rl_checkpoints/reach_avoid_mondrian_shielded_from_scratch",
    )
    parser.add_argument("--hidden-dims", type=str, default="256,256")
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--temp-lr", type=float, default=3e-4)
    parser.add_argument("--lag-lr", type=float, default=3e-4)
    parser.add_argument("--algorithm", choices=["sac", "sac_lag"], default="sac_lag")
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--init-temperature", type=float, default=0.1)
    parser.add_argument("--init-lag", type=float, default=0.0)
    parser.add_argument("--cost-limit", type=float, default=1.0)

    # The true Gymnasium plant and task remain unchanged.
    parser.add_argument("--speed", type=float, default=0.60)
    parser.add_argument("--speed-min", type=float, default=0.0)
    parser.add_argument("--beta-u", type=float, default=0.50)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--initial-x", type=float, default=-2.0)
    parser.add_argument("--initial-y", type=float, default=-2.0)
    parser.add_argument("--initial-theta", type=float, default=math.pi / 4.0)
    parser.add_argument("--start-jitter-xy", type=float, default=0.60)
    parser.add_argument("--start-jitter-theta", type=float, default=0.35)
    parser.add_argument("--dense-safety-cost", dest="dense_safety_cost", action="store_true", default=True)
    parser.add_argument("--no-dense-safety-cost", dest="dense_safety_cost", action="store_false")
    parser.add_argument("--dense-safety-weight", type=float, default=0.25)
    parser.add_argument("--safety-cost-margin", type=float, default=0.10)
    parser.add_argument("--binary-unsafe-cost", action="store_true")
    parser.add_argument(
        "--cost-on-nominal-action",
        action="store_true",
        default=True,
        help=(
            "Train the constraint critic on the actor proposal's one-step "
            "counterfactual safety cost so shield intervention is not hidden."
        ),
    )
    parser.add_argument(
        "--cost-on-executed-action",
        dest="cost_on_nominal_action",
        action="store_false",
        help="Ablation: train the constraint critic on executed-transition cost.",
    )
    parser.add_argument("--goal-bonus", type=float, default=0.0)
    parser.add_argument("--action-penalty", type=float, default=0.0)
    parser.add_argument("--unsafe-penalty", type=float, default=1.0)
    parser.add_argument("--out-of-bounds-penalty", type=float, default=250.0)
    parser.add_argument("--collision-terminal-penalty", type=float, default=200.0)
    parser.add_argument("--terminate-on-unsafe", dest="terminate_on_unsafe", action="store_true", default=True)
    parser.add_argument("--terminate-on-collision", dest="terminate_on_unsafe", action="store_true")

    parser.add_argument(
        "--training-shield-method",
        choices=["nominal", "cbvf", "cp"],
        default="cp",
        help="Action filter used during RL data collection; each setting trains an independent actor.",
    )
    parser.add_argument("--gamma", type=float, default=0.10)
    parser.add_argument("--delta-traj", "--delta", dest="delta_traj", type=float, default=0.05)
    parser.add_argument("--n-calib", type=int, default=500)
    parser.add_argument(
        "--mondrian-clearance-edges",
        type=str,
        default="0.25,0.75,1.50",
    )
    parser.add_argument("--epsilon-grid", type=float, default=0.0)
    parser.add_argument("--epsilon-inter", type=float, default=0.0)
    parser.add_argument(
        "--allow-infeasible-cp-diagnostic-run",
        action="store_true",
        help=(
            "Diagnostic-only: continue with the finalized solver's explicit "
            "least-violation action when the calibrated CP-QP is infeasible. "
            "Strict training aborts before executing such an action by default."
        ),
    )
    parser.add_argument(
        "--qp-diagnostic-episodes",
        type=int,
        default=2,
        help=(
            "Number of CP evaluation episodes per evaluation point for which "
            "every active-step feasibility breakdown is printed and saved."
        ),
    )
    parser.add_argument(
        "--qp-diagnostic-print-limit",
        type=int,
        default=200,
        help="Maximum active CP diagnostic lines printed per evaluation point.",
    )
    parser.add_argument("--speed-envelope", type=float, default=None)
    parser.add_argument(
        "--calibration-stochastic-policy",
        dest="calibration_stochastic_policy",
        action="store_true",
        default=True,
        help="Use the fixed initial stochastic actor during calibration (default).",
    )
    parser.add_argument(
        "--calibration-deterministic-policy",
        dest="calibration_stochastic_policy",
        action="store_false",
    )
    parser.set_defaults(calib_continue_after_unsafe=True)
    parser.add_argument(
        "--no-calib-continue-after-unsafe",
        dest="calib_continue_after_unsafe",
        action="store_false",
    )

    # Approximate model used for the CBVF. Defaults match the true plant, but
    # separate values allow the existing mismatch experiment without changing
    # the environment or the Mondrian mechanism.
    parser.add_argument("--cbvf-model-speed", type=float, default=None)
    parser.add_argument("--cbvf-model-speed-min", type=float, default=None)
    parser.add_argument("--cbvf-model-beta-u", type=float, default=None)
    parser.add_argument("--cbvf-activate-margin", type=float, default=0.0)
    parser.add_argument("--cp-activate-margin", type=float, default=0.10)
    parser.add_argument("--cache-dir", type=str, default="cbvf_cache")
    parser.add_argument(
        "--recompute-cbvf",
        action="store_true",
        default=True,
        help="Accepted for explicit commands; fresh recomputation is mandatory and cannot be disabled.",
    )
    parser.add_argument("--cbvf-nx", type=int, default=81)
    parser.add_argument("--cbvf-ny", type=int, default=81)
    parser.add_argument("--cbvf-nth", type=int, default=61)
    parser.add_argument("--cbvf-dt", type=float, default=0.10)
    parser.add_argument(
        "--time-invariant-cbvf",
        action="store_true",
        help="Compute a converged stationary CBVF, avoiding near-terminal finite-horizon authority loss.",
    )
    parser.add_argument(
        "--cbvf-terminal-guard",
        type=float,
        default=1.0,
        help=(
            "Extra certified lookahead after the last deployed control interval. "
            "This avoids using the terminal target slice as an instantaneous "
            "first-order steering certificate."
        ),
    )
    parser.add_argument("--solver-accuracy", choices=["low", "medium", "high"], default="low")
    parser.add_argument("--target-shape", choices=["signed_distance", "squared"], default="signed_distance")
    parser.add_argument("--target-clip", type=float, default=1.0)
    parser.add_argument("--max-abs-table", type=float, default=10.0)
    parser.add_argument("--max-abs-grad", type=float, default=200.0)
    parser.add_argument("--max-abs-solver-value", type=float, default=50.0)
    parser.add_argument("--max-reasonable-xi", type=float, default=5.0)
    parser.set_defaults(require_active_mondrian=True)
    parser.add_argument(
        "--allow-single-effective-region",
        dest="require_active_mondrian",
        action="store_false",
        help=(
            "Debug/smoke-only override. Full experiments abort if sparse-region "
            "fallback collapses the Mondrian cut to one global region."
        ),
    )
    parser.add_argument("--allow-coarse-fallback", action="store_true")
    parser.add_argument("--max-est-runtime-gb", type=float, default=3.0)
    parser.add_argument("--odp-root", type=str, default=None)
    parser.add_argument("--solver-file", type=str, default=None)

    parser.add_argument("--eval-jsonl", type=str, default=None)
    parser.add_argument("--action-log-jsonl", type=str, default=None)
    parser.add_argument("--qp-diagnostics-jsonl", type=str, default=None)
    parser.add_argument("--results-json", type=str, default=None)
    parser.add_argument("--plot-dir", type=str, default=None)
    parser.add_argument("--action-plot-stride", type=int, default=100)
    parser.add_argument("--keep-if-nominal-unsafe-below", type=float, default=0.20)
    parser.add_argument("--keep-if-goal-above", type=float, default=0.75)
    parser.add_argument("--no-tqdm", action="store_true")
    args = parser.parse_args()

    if args.max_steps < 1:
        parser.error("--max-steps must be positive.")
    if not 1 <= args.start_training <= args.max_steps:
        parser.error("--start-training must be between 1 and --max-steps.")
    if not args.start_training <= args.reference_proposal_steps <= args.max_steps:
        parser.error(
            "--reference-proposal-steps must be between --start-training and --max-steps."
        )
    if not 0.0 <= args.residual_reference_scale <= 1.0:
        parser.error("--residual-reference-scale must lie in [0,1].")
    if args.batch_size < 1 or args.utd_ratio < 1:
        parser.error("--batch-size and --utd-ratio must be positive.")
    if min(args.eval_interval, args.log_interval, args.save_interval) < 1:
        parser.error("Evaluation, logging, and save intervals must be positive.")
    if args.eval_episodes < 1 or args.periodic_eval_episodes < 1 or args.n_calib < 1:
        parser.error("Evaluation episode counts and --n-calib must be positive.")
    if not 0.0 < args.delta_traj < 1.0:
        parser.error("--delta-traj must lie strictly between zero and one.")
    if args.epsilon_grid < 0.0 or args.epsilon_inter < 0.0:
        parser.error("--epsilon-grid and --epsilon-inter must be nonnegative.")
    if args.cbvf_terminal_guard < 0.0:
        parser.error("--cbvf-terminal-guard must be nonnegative.")
    if args.qp_diagnostic_episodes < 0 or args.qp_diagnostic_print_limit < 0:
        parser.error(
            "--qp-diagnostic-episodes and --qp-diagnostic-print-limit "
            "must be nonnegative."
        )
    if args.speed_envelope is not None and args.speed_envelope <= 0.0:
        parser.error("--speed-envelope must be positive when provided.")
    if args.action_plot_stride < 1:
        parser.error("--action-plot-stride must be positive.")
    if args.target_shape != "signed_distance":
        parser.error(
            "Shielded paper training requires --target-shape signed_distance "
            "so the CBVF target and environment safety margin remain consistent."
        )

    args.cbvf_model_speed = (
        float(args.speed) if args.cbvf_model_speed is None else float(args.cbvf_model_speed)
    )
    args.cbvf_model_speed_min = (
        float(args.speed_min)
        if args.cbvf_model_speed_min is None
        else float(args.cbvf_model_speed_min)
    )
    args.cbvf_model_beta_u = (
        float(args.beta_u)
        if args.cbvf_model_beta_u is None
        else float(args.cbvf_model_beta_u)
    )

    hidden_dims = _parse_hidden_dims(args.hidden_dims)
    if not hidden_dims:
        parser.error("--hidden-dims must contain at least one integer.")

    ckpt_dir = Path(args.checkpoint_dir).expanduser().resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    eval_jsonl = (
        Path(args.eval_jsonl).expanduser().resolve()
        if args.eval_jsonl
        else ckpt_dir / "training_eval_metrics.jsonl"
    )
    action_log_path = (
        Path(args.action_log_jsonl).expanduser().resolve()
        if args.action_log_jsonl
        else ckpt_dir / "training_actions.jsonl"
    )
    qp_diagnostics_path = (
        Path(args.qp_diagnostics_jsonl).expanduser().resolve()
        if args.qp_diagnostics_jsonl
        else ckpt_dir / "qp_feasibility_diagnostics.jsonl"
    )
    results_path = (
        Path(args.results_json).expanduser().resolve()
        if args.results_json
        else ckpt_dir / "training_and_evaluation_results.json"
    )
    plot_dir = (
        Path(args.plot_dir).expanduser().resolve()
        if args.plot_dir
        else ckpt_dir / "plots"
    )
    for output_parent in (
        eval_jsonl.parent,
        action_log_path.parent,
        qp_diagnostics_path.parent,
        results_path.parent,
        plot_dir,
    ):
        output_parent.mkdir(parents=True, exist_ok=True)
    eval_jsonl.write_text("", encoding="utf-8")
    qp_diagnostics_path.write_text("", encoding="utf-8")

    env_kwargs = dict(
        speed=args.speed,
        speed_min=args.speed_min,
        beta_u=args.beta_u,
        dt=args.dt,
        horizon=args.horizon,
        initial_state=(args.initial_x, args.initial_y, args.initial_theta),
        random_start=True,
        start_jitter_xy=args.start_jitter_xy,
        start_jitter_theta=args.start_jitter_theta,
        dense_safety_cost=args.dense_safety_cost,
        dense_safety_weight=args.dense_safety_weight,
        safety_cost_margin=args.safety_cost_margin,
        binary_unsafe_cost=args.binary_unsafe_cost,
        goal_bonus=args.goal_bonus,
        action_penalty=args.action_penalty,
        unsafe_penalty=args.unsafe_penalty,
        out_of_bounds_penalty=args.out_of_bounds_penalty,
        collision_terminal_penalty=args.collision_terminal_penalty,
        terminate_on_unsafe=args.terminate_on_unsafe,
    )
    env = gym.make(
        "ConformalDubins3d-v0",
        max_episode_steps=args.horizon,
        **env_kwargs,
    )
    eval_env = gym.make(
        "ConformalDubins3d-v0",
        max_episode_steps=args.horizon,
        **env_kwargs,
    )
    _assert_time_limit_matches_horizon(env, args.horizon)
    _assert_time_limit_matches_horizon(eval_env, args.horizon)

    # No earlier RL checkpoint is restored. The agent is created before
    # calibration only so the fixed initial nominal policy can generate the
    # required baseline-CBVF calibration rollouts.
    agent = _make_agent(args, env, hidden_dims)
    print("Fresh RL agent created; no pretrained checkpoint was restored.")
    print("Rebuilding CBVF, then calibrating Mondrian buffers before episode one.")

    train_shield = TrainingShield(args, env, agent_for_calibration=agent)
    if not train_shield.cbvf_recomputed:
        raise RuntimeError("Training cannot start without fresh CBVF recomputation.")
    if train_shield.mondrian_calibration is None:
        raise RuntimeError("Training cannot start without Mondrian calibration.")

    shield_metadata = train_shield.metadata()
    calibration_manifest = ckpt_dir / "pretraining_mondrian_calibration.json"
    _write_json(calibration_manifest, shield_metadata)
    if (
        args.training_shield_method == "cp"
        and
        not train_shield.offline_feasibility_audit["discrete_grid_gate_passed"]
        and not args.allow_infeasible_cp_diagnostic_run
    ):
        worst = train_shield.offline_feasibility_audit[
            "worst_active_certified_grid_node"
        ]
        raise RuntimeError(
            "The discrete pre-training CP-QP feasibility gate failed before "
            "RL data collection. The calibration manifest contains the full "
            f"audit; worst node={json.dumps(worst, sort_keys=True)}. Adjust the "
            "validated feasibility/envelope design rather than relying on the "
            "least-violation rail fallback. Use "
            "--allow-infeasible-cp-diagnostic-run only for a labeled short diagnosis."
        )
    if (
        args.training_shield_method == "cp"
        and
        args.require_active_mondrian
        and train_shield.mondrian_calibration.n_effective_regions <= 1
    ):
        raise RuntimeError(
            "Mondrian calibration collapsed to one global effective region. "
            "The calibration manifest was saved, but RL training was not started. "
            "Increase calibration coverage/count or revise the fixed reference "
            "rollout design before making a Mondrian-activity claim. Use "
            "--allow-single-effective-region only for an explicitly labeled smoke test."
        )

    policy_config = {
        "seed": int(args.seed),
        "algorithm": args.algorithm,
        "hidden_dims": list(hidden_dims),
        "cost_limit": float(args.cost_limit),
        "init_temperature": float(args.init_temperature),
        "env_kwargs": env_kwargs,
        "training_workflow": {
            "fresh_agent": True,
            "pretrained_checkpoint_restored": False,
            "cbvf_recomputed_before_training": True,
            "calibration_before_first_environment_step": True,
            "calibration_before_first_optimizer_update": True,
            "pretraining_calibration_reference_policy": "fixed_nominal_model_reference_controller",
            "final_calibration_policy": "frozen_reference_plus_learned_residual",
            "calibration_rollout_shield": "baseline_cbvf_qp_xi_0",
            "training_rollout_shield": args.training_shield_method,
            "training_transition_action": "bounded_residual_decision_variable",
            "environment_transition_action": "reference_plus_residual_then_selected_filter",
            "training_transition_cost": "counterfactual_one_step_cost_of_unfiltered_actor_proposal",
            "bootstrap_mask": "1_minus_true_termination_truncation_still_bootstraps",
            "gym_time_limit_equals_cbvf_horizon": True,
            "evaluation_methods_share_identical_seeded_starts": True,
            "nominal_evaluation_role": (
                "independently_trained_nominal_baseline"
                if args.training_shield_method == "nominal"
                else "counterfactual_unshielded_diagnostic_of_a_filtered_actor"
            ),
            "cp_infeasibility_policy": (
                "explicit_diagnostic_least_violation_fallback"
                if args.allow_infeasible_cp_diagnostic_run
                else "strict_abort_before_fallback_execution"
            ),
            "assumption5_claim": (
                "not_claimed_if_any_runtime_feasibility_margin_is_negative"
            ),
            "training_rollout_coverage_claim": "none_policy_changes_after_calibration",
            "post_training_recalibration": False,
            "aci": False,
        },
        "shield": shield_metadata,
        "output_files": {
            "calibration_manifest": str(calibration_manifest),
            "action_log_jsonl": str(action_log_path),
            "qp_feasibility_diagnostics_jsonl": str(qp_diagnostics_path),
            "evaluation_jsonl": str(eval_jsonl),
            "results_json": str(results_path),
            "plot_dir": str(plot_dir),
        },
    }
    _write_json(ckpt_dir / "policy_config.json", policy_config)

    replay_buffer = ReplayBuffer(env.observation_space, env.action_space, args.max_steps)
    replay_buffer.seed(args.seed)

    # This is the first reset for RL data collection. CBVF recomputation and
    # Mondrian calibration have already succeeded above.
    obs, _ = _reset_certified_env(env, train_shield, seed=args.seed)
    done = False
    episode_index = 0
    ep_return = 0.0
    ep_cost = 0.0
    ep_len = 0
    ep_interventions = 0.0
    ep_active = 0.0
    ep_infeasible = 0.0
    best_saved_step: Optional[int] = None
    eval_records: List[Dict[str, Any]] = []
    action_samples: List[Dict[str, Any]] = []
    training_candidate_region_count_sum = 0
    training_candidate_region_count_max = 0
    training_candidate_regions_seen: set = set()
    training_active_cp_steps = 0
    training_infeasible_cp_steps = 0
    training_rail_fallback_steps = 0
    training_no_authority_fallback_steps = 0
    training_feasibility_margin_sum = 0.0
    training_feasibility_margin_min = float("inf")
    training_feasibility_margin_max = -float("inf")

    iterator = tqdm.tqdm(
        range(1, args.max_steps + 1),
        disable=args.no_tqdm,
        smoothing=0.1,
    )
    action_log_file = action_log_path.open("w", encoding="utf-8")
    try:
        for step in iterator:
            state_before = env.unwrapped.state.copy()
            # Reference warmup supplies goal-reaching off-policy transitions;
            # its proposals still pass through the selected training shield.
            if step <= args.reference_proposal_steps and args.warmup_policy == "reference":
                policy_action = np.zeros(2, dtype=np.float32)
                u_nom = _reference_warmup_action(
                    state_before, args.cbvf_model_beta_u
                )
            else:
                action, agent = agent.sample_actions(obs)
                residual = np.asarray(action, dtype=np.float32).reshape(2)
                policy_action = residual
                reference = _reference_warmup_action(
                    state_before, args.cbvf_model_beta_u
                )
                u_nom = np.clip(
                    reference + args.residual_reference_scale * residual,
                    -1.0,
                    1.0,
                ).astype(np.float32)
            u_filtered, shield_diag = train_shield.project(
                state_before,
                ep_len,
                u_nom,
                method=args.training_shield_method,
            )
            step_candidate_regions = tuple(
                int(value)
                for value in shield_diag["candidate_effective_regions"]
            )
            training_candidate_region_count_sum += len(step_candidate_regions)
            training_candidate_region_count_max = max(
                training_candidate_region_count_max,
                len(step_candidate_regions),
            )
            training_candidate_regions_seen.update(step_candidate_regions)
            if bool(shield_diag["active"]):
                training_active_cp_steps += 1
                feasibility_margin = float(shield_diag["feasibility_margin"])
                training_feasibility_margin_sum += feasibility_margin
                training_feasibility_margin_min = min(
                    training_feasibility_margin_min,
                    feasibility_margin,
                )
                training_feasibility_margin_max = max(
                    training_feasibility_margin_max,
                    feasibility_margin,
                )
                training_infeasible_cp_steps += int(
                    bool(shield_diag["infeasible"])
                )
                training_rail_fallback_steps += int(
                    bool(shield_diag["fallback_is_actuator_rail"])
                )
                training_no_authority_fallback_steps += int(
                    shield_diag["fallback_kind"]
                    == "no_control_authority_keep_nominal"
                )

            next_obs, reward, terminated, truncated, info = env.step(
                np.asarray(u_filtered, dtype=np.float32)
            )
            done = bool(terminated or truncated)
            environment_action = np.asarray(info["applied_action"], dtype=float)
            if not np.allclose(environment_action, u_filtered, rtol=0.0, atol=1e-6):
                raise RuntimeError(
                    "Environment action mismatch: the transition was not generated "
                    "by the filtered action recorded in the replay tuple."
                )

            mask = float(not terminated)
            executed_cost = float(info.get("cost", 0.0))
            nominal_counterfactual_cost, nominal_cost_info = (
                env.unwrapped.cost_for_state_action(state_before, u_nom)
            )
            learning_cost = (
                float(nominal_counterfactual_cost)
                if args.cost_on_nominal_action
                else executed_cost
            )

            # Reward and next observation describe the executed transition.
            # The constraint critic can instead score the actor proposal; this
            # intervention-aware signal prevents a safety filter from hiding
            # unsafe nominal actions during shielded learning.
            replay_buffer.insert(
                dict(
                    observations=obs,
                    # The learned MDP action is the bounded residual. The
                    # deterministic reference composition and safety filter
                    # are part of the environment transition map.
                    actions=np.asarray(policy_action, dtype=np.float32),
                    rewards=float(reward),
                    costs=learning_cost,
                    masks=mask,
                    dones=done,
                    next_observations=next_obs,
                )
            )

            action_record = {
                "step": int(step),
                "episode": int(episode_index),
                "episode_step": int(ep_len),
                "nominal_action": [float(value) for value in u_nom],
                "filtered_action": [float(value) for value in u_filtered],
                "environment_applied_action": [float(value) for value in environment_action],
                "transition_action_matches_environment": True,
                "bootstrap_mask": float(mask),
                "intervened": bool(shield_diag["intervention"]),
                "shield_active": bool(shield_diag["active"]),
                "qp_infeasible": bool(shield_diag["infeasible"]),
                "assumption5_violation": bool(shield_diag["assumption5_violation"]),
                "B": float(shield_diag["B"]),
                "regional_buffer": float(shield_diag["regional_buffer"]),
                "qp_rhs": float(shield_diag["qp_rhs"]),
                "xi_hat_app": float(shield_diag["xi_hat_app"]),
                "epsilon_inter": float(shield_diag["epsilon_inter"]),
                "requested_xi": float(shield_diag["requested_xi"]),
                "max_u_psi": shield_diag["max_u_psi"],
                "feasibility_margin": shield_diag["feasibility_margin"],
                "achieved_lhs": shield_diag["achieved_lhs"],
                "achieved_constraint_margin": shield_diag[
                    "achieved_constraint_margin"
                ],
                "a_u": shield_diag["a_u"],
                "base_without_control": shield_diag["base_without_control"],
                "fallback_used": bool(shield_diag["fallback_used"]),
                "fallback_kind": str(shield_diag["fallback_kind"]),
                "fallback_is_actuator_rail": bool(
                    shield_diag["fallback_is_actuator_rail"]
                ),
                "fallback_bound": str(shield_diag["fallback_bound"]),
                "minimal_projection_verified": shield_diag[
                    "minimal_projection_verified"
                ],
                "candidate_effective_regions": [
                    int(v) for v in step_candidate_regions
                ],
                "reward": float(reward),
                "executed_transition_cost": float(executed_cost),
                "learning_constraint_cost": float(learning_cost),
                "nominal_counterfactual_cost": float(nominal_counterfactual_cost),
                "nominal_counterfactual_safe_margin": float(
                    nominal_cost_info["safe_margin"]
                ),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
            action_log_file.write(json.dumps(action_record, sort_keys=True) + "\n")
            if (
                step % args.action_plot_stride == 0
                or bool(shield_diag["intervention"])
                or bool(shield_diag["fallback_used"])
            ):
                action_samples.append({
                    "step": int(step),
                    "nominal_action": [float(value) for value in u_nom],
                    "filtered_action": [float(value) for value in u_filtered],
                    "fallback_used": bool(shield_diag["fallback_used"]),
                    "feasible": bool(not shield_diag["infeasible"]),
                })

            obs = next_obs
            ep_return += float(reward)
            ep_cost += executed_cost
            ep_len += 1
            ep_interventions += float(shield_diag["intervention"])
            ep_active += float(shield_diag["active"])
            ep_infeasible += float(shield_diag["infeasible"])

            if done:
                if step % args.log_interval == 0:
                    print(
                        f"step={step} train_return={ep_return:.2f} "
                        f"train_executed_cost={ep_cost:.2f} train_len={ep_len} "
                        f"cp_interventions={ep_interventions:.1f} "
                        f"cp_active={ep_active:.1f} cp_infeasible={ep_infeasible:.1f}"
                    )
                episode_index += 1
                obs, _ = _reset_certified_env(env, train_shield)
                done = False
                ep_return = 0.0
                ep_cost = 0.0
                ep_len = 0
                ep_interventions = 0.0
                ep_active = 0.0
                ep_infeasible = 0.0

            if step >= args.start_training:
                batch = replay_buffer.sample(args.batch_size * args.utd_ratio)
                agent, update_info = agent.update(batch, args.utd_ratio)
                if step % args.log_interval == 0:
                    printable = {
                        key: float(np.asarray(value).mean())
                        for key, value in update_info.items()
                        if np.asarray(value).size > 0
                    }
                    iterator.set_postfix(printable)

            if step % args.eval_interval == 0:
                stats_by_method = evaluate_all(
                    agent,
                    eval_env,
                    episodes=args.periodic_eval_episodes,
                    seed=args.seed + 10_000 + step,
                    shield=train_shield,
                    evaluation_step=step,
                    qp_diagnostic_episodes=args.qp_diagnostic_episodes,
                    qp_diagnostic_print_limit=args.qp_diagnostic_print_limit,
                    qp_diagnostics_path=qp_diagnostics_path,
                )
                msg = [f"eval step={step}"]
                for method, stats in stats_by_method.items():
                    msg.append(
                        f"{method}: unsafe={stats['unsafe_rate']:.3f} "
                        f"goal={stats['goal_rate']:.3f} ret={stats['return']:.2f} "
                        f"cost={stats['cost']:.2f} int={stats['interventions']:.1f} "
                        f"infeas={stats['infeasible_steps']:.1f}"
                    )
                print(" | ".join(msg))
                cp_stats = stats_by_method["cp"]
                print(
                    "CP-QP feasibility summary: "
                    f"active={int(cp_stats['active_filter_total'])}, "
                    f"infeasible={int(cp_stats['infeasible_total'])}, "
                    f"infeasible/active={cp_stats['infeasible_rate_among_active']:.3f}, "
                    f"rail_fallbacks={int(cp_stats['rail_fallback_total'])}, "
                    f"xi_hat_app_mean={cp_stats['mean_active_xi_hat_app']:.6f}, "
                    f"epsilon_inter={cp_stats['epsilon_inter']:.6f}, "
                    f"max_u_Psi_mean={cp_stats['mean_active_max_u_psi']:.6f}, "
                    f"feasibility_margin_min={cp_stats['min_active_feasibility_margin']:.6f}, "
                    f"minimal_projection_failures="
                    f"{int(cp_stats['minimal_projection_failure_total'])}"
                )
                eval_record = {
                    "step": int(step),
                    "methods": stats_by_method,
                    "training_shield_calibration": "pretraining_fixed_buffers",
                    "coverage_claim_for_training_policy": "none_for_changing_training_policy",
                    "shared_initial_state_seeds_across_methods": True,
                    "method_semantics": {
                        "nominal": (
                            "primary independently trained method"
                            if args.training_shield_method == "nominal"
                            else "counterfactual unshielded evaluation"
                        ),
                        "cbvf": (
                            "primary independently trained method"
                            if args.training_shield_method == "cbvf"
                            else "counterfactual baseline-filter evaluation"
                        ),
                        "cp": (
                            "primary independently trained method"
                            if args.training_shield_method == "cp"
                            else "counterfactual calibrated-filter evaluation"
                        ),
                    },
                }
                eval_records.append(eval_record)
                with eval_jsonl.open("a", encoding="utf-8") as eval_file:
                    eval_file.write(json.dumps(eval_record, sort_keys=True) + "\n")

                nominal = stats_by_method["nominal"]
                if (
                    nominal["unsafe_rate"] <= args.keep_if_nominal_unsafe_below
                    and nominal["goal_rate"] >= args.keep_if_goal_above
                    and best_saved_step is None
                ):
                    save_agent(agent, str(ckpt_dir), step)
                    best_saved_step = int(step)
                    (ckpt_dir / "best_checkpoint.txt").write_text(
                        f"params_{step}.pkl\n"
                        f"nominal_unsafe_rate={nominal['unsafe_rate']:.6f}\n"
                        f"nominal_goal_rate={nominal['goal_rate']:.6f}\n",
                        encoding="utf-8",
                    )

            if step % args.save_interval == 0:
                save_agent(agent, str(ckpt_dir), step)

            if step % args.log_interval == 0:
                action_log_file.flush()
    finally:
        action_log_file.close()

    final_step = int(args.max_steps)
    save_agent(agent, str(ckpt_dir), final_step)

    # Freeze the learned residual policy, then perform a new split-conformal
    # calibration on data collected by exactly that deployed policy. No policy
    # update occurs after this point; final evaluation uses disjoint seeds.
    args.post_training_recalibration = True
    final_shield = TrainingShield(args, env, agent)
    if final_shield.offline_feasibility_audit["n_infeasible_grid_nodes"] != 0:
        raise RuntimeError(
            "Post-training calibrated QP failed the exhaustive offline feasibility gate."
        )
    calibration_manifest = ckpt_dir / "final_frozen_policy_mondrian_calibration.json"
    shield_metadata = final_shield.metadata()
    shield_metadata["calibration_phase"] = "post_training_frozen_residual_policy"
    shield_metadata["no_policy_updates_after_calibration"] = True
    _write_json(calibration_manifest, shield_metadata)
    policy_config["shield"] = shield_metadata
    policy_config["training_workflow"]["post_training_recalibration"] = True
    policy_config["training_workflow"]["training_rollout_coverage_claim"] = "none"
    policy_config["training_workflow"]["final_evaluation_coverage_claim"] = (
        "split_conformal_for_frozen_policy_with_disjoint_evaluation_seeds"
    )
    _write_json(ckpt_dir / "policy_config.json", policy_config)

    final_stats = evaluate_all(
            agent,
            eval_env,
            episodes=args.eval_episodes,
            seed=args.seed + 90_000 + final_step,
            shield=final_shield,
            evaluation_step=final_step,
            qp_diagnostic_episodes=args.qp_diagnostic_episodes,
            qp_diagnostic_print_limit=args.qp_diagnostic_print_limit,
            qp_diagnostics_path=qp_diagnostics_path,
            capture_trajectory=True,
        )
    final_record = {
            "step": final_step,
            "methods": final_stats,
        "training_shield_calibration": "post_training_frozen_policy_buffers",
        "coverage_claim_for_training_policy": "split_conformal_frozen_final_policy",
            "shared_initial_state_seeds_across_methods": True,
            "method_semantics": {
                "nominal": (
                    "primary independently trained method"
                    if args.training_shield_method == "nominal"
                    else "counterfactual unshielded evaluation"
                ),
                "cbvf": (
                    "primary independently trained method"
                    if args.training_shield_method == "cbvf"
                    else "counterfactual baseline-filter evaluation"
                ),
                "cp": (
                    "primary independently trained method"
                    if args.training_shield_method == "cp"
                    else "counterfactual calibrated-filter evaluation"
                ),
            },
    }
    eval_records.append(final_record)
    with eval_jsonl.open("a", encoding="utf-8") as eval_file:
        eval_file.write(json.dumps(final_record, sort_keys=True) + "\n")

    generated_plots = _plot_training_outputs(
        eval_records=eval_records,
        action_samples=action_samples,
        plot_dir=plot_dir,
    )
    checkpoint_files = sorted(
        str(path) for path in ckpt_dir.glob("params_*.pkl")
    )
    training_mondrian_activity = {
        **train_shield.mondrian_activity,
        "mean_candidate_effective_regions": float(
            training_candidate_region_count_sum / max(1, final_step)
        ),
        "max_candidate_effective_regions": int(training_candidate_region_count_max),
        "candidate_effective_regions_seen": sorted(
            int(value) for value in training_candidate_regions_seen
        ),
        "n_unique_candidate_effective_regions_seen": int(
            len(training_candidate_regions_seen)
        ),
        "all_effective_regions_seen_during_training": bool(
            len(training_candidate_regions_seen)
            == train_shield.mondrian_calibration.n_effective_regions
        ),
    }
    if training_active_cp_steps > 0:
        training_min_feasibility_margin: Optional[float] = float(
            training_feasibility_margin_min
        )
        training_max_feasibility_margin: Optional[float] = float(
            training_feasibility_margin_max
        )
        training_mean_feasibility_margin: Optional[float] = float(
            training_feasibility_margin_sum / training_active_cp_steps
        )
    else:
        training_min_feasibility_margin = None
        training_max_feasibility_margin = None
        training_mean_feasibility_margin = None
    training_qp_feasibility = {
        "active_cp_steps": int(training_active_cp_steps),
        "infeasible_cp_steps": int(training_infeasible_cp_steps),
        "infeasible_rate_among_active": float(
            training_infeasible_cp_steps / max(1, training_active_cp_steps)
        ),
        "rail_fallback_steps": int(training_rail_fallback_steps),
        "rail_fallback_rate_among_infeasible": float(
            training_rail_fallback_steps / max(1, training_infeasible_cp_steps)
        ),
        "no_control_authority_fallback_steps": int(
            training_no_authority_fallback_steps
        ),
        "min_feasibility_margin": training_min_feasibility_margin,
        "mean_feasibility_margin": training_mean_feasibility_margin,
        "max_feasibility_margin": training_max_feasibility_margin,
        "epsilon_inter": float(train_shield.epsilon_inter),
        "assumption5_runtime_satisfied_on_active_training_states": bool(
            training_infeasible_cp_steps == 0
        ),
        "fallback_policy": (
            "explicit_diagnostic_least_violation_fallback"
            if args.allow_infeasible_cp_diagnostic_run
            else "strict_abort_before_fallback_execution"
        ),
    }
    evaluation_infeasible_cp_steps = int(sum(
        float(record["methods"]["cp"]["infeasible_total"])
        for record in eval_records
    ))
    evaluation_rail_fallback_steps = int(sum(
        float(record["methods"]["cp"]["rail_fallback_total"])
        for record in eval_records
    ))
    training_qp_feasibility.update({
        "evaluation_infeasible_cp_steps_across_recorded_evaluations": (
            evaluation_infeasible_cp_steps
        ),
        "evaluation_rail_fallback_steps_across_recorded_evaluations": (
            evaluation_rail_fallback_steps
        ),
        "assumption5_runtime_satisfied_on_all_recorded_active_states": bool(
            training_infeasible_cp_steps == 0
            and evaluation_infeasible_cp_steps == 0
        ),
    })
    offline_grid_gate_failed = bool(
        not final_shield.offline_feasibility_audit["discrete_grid_gate_passed"]
    )
    if (
        offline_grid_gate_failed
        or training_infeasible_cp_steps > 0
        or evaluation_infeasible_cp_steps > 0
    ):
        result_status = "diagnostic_completed_with_cp_feasibility_failure"
        result_validity = (
            "not_a_validated_calibrated_cbvf_qp_result_"
            "offline_or_runtime_assumption5_gate_failed"
        )
    else:
        result_status = "completed"
        result_validity = "no_cp_infeasibility_observed_on_executed_training_states"
    final_results = {
        "status": result_status,
        "result_validity": result_validity,
        "final_step": final_step,
        "final_evaluation": final_stats,
        "evaluation_records": eval_records,
        "workflow_invariants": policy_config["training_workflow"],
        "cbvf": {
            "recomputed": bool(final_shield.cbvf_recomputed),
            "path": str(final_shield.cbvf_path),
        },
        "mondrian_calibration": final_shield.mondrian_calibration.as_dict(),
        "calibration_stats": final_shield.calibration_stats,
        "mondrian_activity_diagnostics": training_mondrian_activity,
        "offline_cp_qp_feasibility_audit": (
            final_shield.offline_feasibility_audit
        ),
        "training_qp_feasibility_diagnostics": training_qp_feasibility,
        "coverage_statement": {
            "final_frozen_policy": "covered_by_post_training_mondrian_split_conformal_statement",
            "training_policy_iterates": "no_coverage_claim",
            "evaluation_seed_separation": "final_evaluation_seeds_disjoint_from_final_calibration_seeds",
        },
        "generated_checkpoints": checkpoint_files,
        "generated_plots": generated_plots,
        "action_log_jsonl": str(action_log_path),
        "qp_feasibility_diagnostics_jsonl": str(qp_diagnostics_path),
        "evaluation_jsonl": str(eval_jsonl),
        "policy_config": str(ckpt_dir / "policy_config.json"),
        "calibration_manifest": str(calibration_manifest),
    }
    _write_json(results_path, final_results)

    print(f"Training complete. Final checkpoint: {ckpt_dir / f'params_{final_step}.pkl'}")
    if result_status != "completed":
        print(
            "DIAGNOSTIC RESULT ONLY: the offline grid audit or a runtime "
            "calibrated CP-QP feasibility check failed. Any executed "
            "infeasible action used the explicitly enabled least-violation "
            "fallback. Do not present this run as calibrated-QP validation."
        )
    print(f"Final evaluation: {json.dumps(final_stats, sort_keys=True)}")
    print(f"Action log: {action_log_path}")
    print(f"Results JSON: {results_path}")
    print(f"Generated plots: {generated_plots}")


if __name__ == "__main__":
    main()
