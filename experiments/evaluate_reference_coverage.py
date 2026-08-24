#!/usr/bin/env python3
"""Evaluate saved policies on the held-out reference distribution.

The split-conformal proposition covers future rollouts of the same frozen
policy behind the *uncalibrated* CBVF-QP used during calibration.  Calibrated
deployment rollouts have a different closed-loop distribution and are saved as
a separate diagnostic by the trainer.  This script evaluates the proposition's
population without retraining and never overwrites primary training results.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym
import numpy as np

import conformal_shield as cs
import redexp  # noqa: F401
from jaxrl5.agents.agent import restore_agent
from train.train_sac_lag import AgentPolicyCallable, _make_agent


DEFAULTS = {
    "hidden_dims": "256,256", "actor_lr": 3e-4, "critic_lr": 3e-4,
    "temp_lr": 3e-4, "lag_lr": 3e-4, "discount": .99, "tau": .005,
    "init_temperature": .1, "init_lag": 0., "utd_ratio": 1,
    "initial_x": -2., "initial_y": -2., "initial_theta": math.pi / 4,
    "start_jitter_xy": .6, "start_jitter_theta": .35,
    "dense_safety_cost": True, "dense_safety_weight": .25,
    "safety_cost_margin": .1,
    "binary_unsafe_cost": False, "action_penalty": 0.,
    "out_of_bounds_penalty": 250., "terminate_on_unsafe": True,
    "delta_traj": .05, "epsilon_grid": 0., "speed_envelope": None,
    "calibration_stochastic_policy": True, "calib_continue_after_unsafe": True,
    "allow_infeasible_cp_diagnostic_run": False, "cache_dir": "cbvf_cache",
    "solver_accuracy": "low", "target_clip": 1., "max_abs_table": 10.,
    "max_abs_grad": 200., "max_abs_solver_value": 50.,
    "max_reasonable_xi": 5., "allow_coarse_fallback": False,
    "max_est_runtime_gb": 3., "time_invariant_cbvf": False,
    "odp_root": str(ROOT), "solver_file": str(ROOT / "solver_cbvf.py"),
}


def load_args(run_dir: Path, truth_speed_min: float | None) -> SimpleNamespace:
    batch = json.loads((run_dir / "batch_configuration.json").read_text())
    values = dict(DEFAULTS)
    values.update(batch["configuration"])
    values["delta_traj"] = float(values.pop("delta", values.get("delta_traj", .05)))
    if truth_speed_min is not None:
        values["speed_min"] = float(truth_speed_min)
    values["post_training_recalibration"] = True
    return SimpleNamespace(**values)


def make_env(args: SimpleNamespace):
    return gym.make(
        "ConformalDubins3d-v0", max_episode_steps=int(args.horizon),
        speed=args.speed, speed_min=args.speed_min, beta_u=args.beta_u,
        dt=args.dt, horizon=args.horizon,
        initial_state=(args.initial_x, args.initial_y, args.initial_theta),
        random_start=True, start_jitter_xy=args.start_jitter_xy,
        start_jitter_theta=args.start_jitter_theta,
        dense_safety_cost=args.dense_safety_cost,
        dense_safety_weight=args.dense_safety_weight,
        safety_cost_margin=args.safety_cost_margin,
        binary_unsafe_cost=args.binary_unsafe_cost,
        goal_bonus=args.goal_bonus, action_penalty=args.action_penalty,
        unsafe_penalty=args.unsafe_penalty,
        out_of_bounds_penalty=args.out_of_bounds_penalty,
        collision_terminal_penalty=args.collision_terminal_penalty,
        terminate_on_unsafe=args.terminate_on_unsafe,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--truth-speed-min", type=float)
    parser.add_argument("--model-speed-min", type=float)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--residual-scale", type=float)
    parser.add_argument("--fresh-calibration", action="store_true")
    parser.add_argument("--evaluate-deployment", action="store_true")
    parser.add_argument("--output", type=Path)
    ns = parser.parse_args()
    run_dir = ns.run_dir.resolve()
    args = load_args(run_dir, ns.truth_speed_min)
    if ns.model_speed_min is not None:
        args.cbvf_model_speed_min = float(ns.model_speed_min)
    if ns.horizon is not None:
        args.horizon = int(ns.horizon)
    if ns.residual_scale is not None:
        args.residual_reference_scale = float(ns.residual_scale)
    env = make_env(args)
    agent = _make_agent(args, env, tuple(int(x) for x in args.hidden_dims.split(",")))
    agent = restore_agent(agent, str(run_dir), int(args.max_steps))
    cfg = env.unwrapped.cfg
    world = cs.World(
        xmin=cfg.xmin, xmax=cfg.xmax, ymin=cfg.ymin, ymax=cfg.ymax,
        goal=tuple(cfg.goal), goal_radius=cfg.goal_radius,
        obstacle_center=tuple(cfg.obstacle_center),
        obstacle_radius=cfg.obstacle_radius, robot_radius=cfg.robot_radius,
        init_x_range=(args.initial_x, args.initial_x),
        init_y_range=(args.initial_y, args.initial_y),
        init_theta_range=(args.initial_theta, args.initial_theta),
    )
    bounds = cs.Interval(float(cfg.u_min), float(cfg.u_max))
    variant = cs.DynamicsVariant(
        name="HeldOutReferenceEvaluation",
        model=cs.DynamicsSpec(args.cbvf_model_speed, args.cbvf_model_beta_u,
                              args.cbvf_model_speed_min),
        truth=cs.DynamicsSpec(args.speed, args.beta_u, args.speed_min),
    )
    primary = json.loads((run_dir / "training_and_evaluation_results.json").read_text())
    if ns.model_speed_min is None and ns.horizon is None:
        cbvf = cs.CBVFTable.from_npz(primary["cbvf"]["path"],
                                     max_abs_table=args.max_abs_table,
                                     max_abs_grad=args.max_abs_grad)
    else:
        cbvf, _, _ = cs.prepare_cbvf_table(
            cache_dir=ROOT / "cbvf_cache" / "diagnostics", world=world,
            control_bounds=bounds, model_spec=variant.model, gamma=args.gamma,
            dt=args.dt,
            horizon=int(math.ceil((args.horizon * args.dt + args.cbvf_terminal_guard) / args.dt)),
            cbvf_dt=args.cbvf_dt, nx=args.cbvf_nx, ny=args.cbvf_ny,
            nth=args.cbvf_nth, accuracy=args.solver_accuracy, recompute=True,
            odp_root=args.odp_root, solver_file=args.solver_file,
            target_shape=args.target_shape, target_clip=args.target_clip,
            target_scale=0., max_abs_table=args.max_abs_table,
            max_abs_grad=args.max_abs_grad,
            max_abs_solver_value=args.max_abs_solver_value,
            allow_coarse_fallback=False, max_est_runtime_gb=args.max_est_runtime_gb,
            use_time_invariant_cbvf=False,
        )
    tau = np.linspace(-(args.horizon * args.dt + args.cbvf_terminal_guard),
                      -args.cbvf_terminal_guard, args.horizon + 1)
    partition = cs.MondrianPartition(
        clearance_edges=cs.parse_strictly_increasing_floats(
            args.mondrian_clearance_edges, "--mondrian-clearance-edges"),
        obstacle_center=world.obstacle_center,
        inflated_obstacle_radius=world.obstacle_radius + world.robot_radius,
    )
    stored = json.loads(
        (run_dir / "final_frozen_policy_mondrian_calibration.json").read_text()
    )["mondrian_calibration"]
    region_rows = stored["regions"]
    calibration = cs.MondrianCalibration(
        partition=partition,
        effective_groups=tuple(
            tuple(int(v) for v in row["base_regions"]) for row in region_rows
        ),
        base_to_effective=tuple(int(v) for v in stored["base_to_effective"]),
        buffers=tuple(float(row["xi_hat_off"]) for row in region_rows),
        deltas=tuple(float(row["delta_m"]) for row in region_rows),
        counts=tuple(int(row["n_scores"]) for row in region_rows),
        delta_traj=float(stored["delta_traj"]),
        epsilon_grid=float(stored["epsilon_grid"]),
        promotion_rounds=int(stored["promotion_rounds"]),
    )
    policy = AgentPolicyCallable(
        agent, tuple(cfg.goal), bounds.lo, bounds.hi, variant.model.beta_u,
        residual_scale=args.residual_reference_scale,
        use_frozen_residual=True, stochastic=args.calibration_stochastic_policy,
    )
    rng = np.random.default_rng(int(args.seed + 120_000 + args.max_steps))
    starts = [env.unwrapped.sample_initial_state(rng) for _ in range(ns.episodes)]
    calibration_rng = np.random.default_rng(int(args.seed + 30_000))
    calibration_starts = [
        env.unwrapped.sample_initial_state(calibration_rng)
        for _ in range(int(args.n_calib))
    ]
    if ns.fresh_calibration:
        calibration, _ = cs.calibrate_mondrian(
            world=world, control_bounds=bounds, variant=variant, cbvf=cbvf,
            policy=policy, tau=tau, init_states=calibration_starts,
            gamma=args.gamma, delta_traj=args.delta_traj,
            shield_cfg=cs.ShieldConfig(args.cbvf_activate_margin,
                                       args.cp_activate_margin),
            continue_after_unsafe=args.calib_continue_after_unsafe,
            partition=partition, epsilon_grid=args.epsilon_grid,
        )
    split = cs.assert_disjoint_initial_state_arrays(
        calibration_starts, starts
    )
    logs = cs.evaluate_method(
        world=world, control_bounds=bounds, variant=variant, cbvf=cbvf,
        policy=policy, tau=tau, init_states=starts, method="cbvf", xi=0.,
        gamma=args.gamma,
        shield_cfg=cs.ShieldConfig(args.cbvf_activate_margin,
                                   args.cp_activate_margin),
        mondrian_calibration=calibration,
        speed_envelope=max(math.hypot(spec.v, abs(spec.beta_u))
                           for spec in (variant.model, variant.truth)),
        epsilon_inter=0.,
    )
    summary = cs.summarize(logs)
    deployment_logs = cs.evaluate_method(
        world=world, control_bounds=bounds, variant=variant, cbvf=cbvf,
        policy=policy, tau=tau, init_states=starts, method="cp", xi=0.,
        gamma=args.gamma,
        shield_cfg=cs.ShieldConfig(args.cbvf_activate_margin,
                                   args.cp_activate_margin),
        mondrian_calibration=calibration,
        speed_envelope=max(math.hypot(spec.v, abs(spec.beta_u))
                           for spec in (variant.model, variant.truth)),
        epsilon_inter=args.epsilon_inter,
    ) if ns.evaluate_deployment else []
    deployment_summary = cs.summarize(deployment_logs) if deployment_logs else None
    payload = {
        "run_id": run_dir.name, "episodes": int(ns.episodes),
        "evaluation_population": "held_out_uncalibrated_cbvf_reference_closed_loop",
        "calibration_population": "uncalibrated_cbvf_reference_closed_loop",
        "truth_speed_min_override": ns.truth_speed_min,
        "model_speed_min_override": ns.model_speed_min,
        "horizon_override": ns.horizon,
        "residual_scale_override": ns.residual_scale,
        "target_simultaneous_coverage": 1. - float(args.delta_traj),
        "empirical_simultaneous_coverage": summary["simultaneous_buffer_coverage"],
        "unsafe_rate": summary["unsafe_rate"], "goal_rate": summary["goal_rate"],
        "mean_return": summary["mean_return"],
        "disjointness": split,
        "fresh_calibration": bool(ns.fresh_calibration),
        "calibrated_deployment_summary": deployment_summary,
        "offline_feasibility_audit": primary["offline_cp_qp_feasibility_audit"],
        "calibration": calibration.as_dict(),
        "raw_rollouts": [
            {k: v for k, v in rollout.items() if not k.startswith("_")}
            for rollout in logs
        ],
    }
    output = ns.output or (run_dir / "heldout_reference_coverage.json")
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k not in ("raw_rollouts", "offline_feasibility_audit", "calibration")}, indent=2))


if __name__ == "__main__":
    main()
