#!/usr/bin/env python3
"""Dependency-free static invariants for the shielded training workflow."""

from __future__ import annotations

import ast
import argparse
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_FILE = PROJECT_ROOT / "train" / "train_sac_lag.py"
ENV_FILE = PROJECT_ROOT / "redexp" / "envs" / "conformal_dubins_3d_env.py"
REGISTRATION_FILE = PROJECT_ROOT / "redexp" / "__init__.py"
SHIELD_FILE = PROJECT_ROOT / "conformal_shield.py"
FINALIZED_SHIELD_SHA256 = (
    "ff6e0cc5e6d38060481b992e792fa549955d0dbdacba05c8a2af92bca396cf15"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--shield-sha256', default=FINALIZED_SHIELD_SHA256,
                        help='Expected shield hash from a recorded/frozen configuration; default retains the original supplied-file check.')
    parser.add_argument('--configuration', type=Path,
                        help='Recorded batch configuration or frozen manifest for the configurable recomputation path.')
    args = parser.parse_args()
    training = TRAINING_FILE.read_text(encoding="utf-8")
    environment = ENV_FILE.read_text(encoding="utf-8")
    registration = REGISTRATION_FILE.read_text(encoding="utf-8")
    shield = SHIELD_FILE.read_text(encoding="utf-8")
    ast.parse(training)
    ast.parse(environment)
    ast.parse(registration)
    ast.parse(shield)

    shield_hash = hashlib.sha256(SHIELD_FILE.read_bytes()).hexdigest()
    require(
        shield_hash == args.shield_sha256,
        "conformal_shield.py differs from the expected recorded/frozen source",
    )
    if "recompute=True" not in training:
        require(args.configuration is not None, 'Configurable CBVF recomputation requires an explicit recorded configuration')
        record = json.loads(args.configuration.read_text())
        configuration = record.get('configuration', record.get('common', {}))
        require(configuration.get('recompute_cbvf') is True and not configuration.get('reuse_cbvf', False),
                'Recorded configuration does not require a fresh CBVF solve')
        require('recompute=bool(args.recompute_cbvf)' in training and
                'if bool(args.recompute_cbvf) and not self.cbvf_recomputed:' in training,
                'Fresh-recomputation configuration is not enforced at runtime')
    require("cs.calibrate_mondrian(" in training, "Mondrian calibration is missing")
    require(
        'choices=["nominal", "cbvf", "cp"]' in training
        and "method=args.training_shield_method" in training,
        "independent nominal/CBVF/CP training modes are not wired to action execution",
    )
    require("restore_agent" not in training, "a pretrained-policy path remains")
    require("calibrate_cp_one_shot" not in training, "obsolete global calibration remains")
    require(
        "actions=np.asarray(policy_action" in training
        and "residual_reference_scale * residual" in training,
        "bounded residual RL action is not wired through reference composition",
    )
    require(
        "costs=learning_cost" in training
        and "nominal_counterfactual_cost" in training
        and "cost_on_nominal_action" in training,
        "intervention-aware proposal cost is not wired to the constraint critic",
    )
    require("env.step(\n                np.asarray(u_filtered" in training, "environment does not execute the filtered action")
    require("nominal_action" in training and "filtered_action" in training, "dual action logging is missing")
    require("method=\"cbvf\"" in shield, "finalized baseline-CBVF calibration call is missing")
    require("xi=0.0" in shield, "baseline calibration is not explicitly xi=0")
    require(
        "mask = float(not terminated)" in training,
        "bootstrap mask is not exactly 1 - terminated",
    )
    require(
        "not terminated or truncated" not in training,
        "the terminated/truncated precedence bug remains",
    )
    require(
        training.count("max_episode_steps=args.horizon") == 2,
        "train/eval Gym TimeLimits are not both driven by --horizon",
    )
    require(
        "max_episode_steps=400" not in registration,
        "environment registration still hardcodes a 400-step TimeLimit",
    )
    require(
        training.count("_assert_time_limit_matches_horizon(") >= 3,
        "the Gym/internal/CBVF horizon invariant is not asserted",
    )
    require(
        "def sample_initial_state" in environment
        and "env.unwrapped.sample_initial_state(calibration_rng)" in training,
        "calibration and RL do not share the environment start sampler",
    )
    require(
        "init_x_range=(self.initial_state[0], self.initial_state[0])" in training,
        "the calibration World start is not derived from environment config",
    )
    require(
        "seed + i * 1000" not in training,
        "evaluation methods still use different start-state seeds",
    )
    require(
        "evaluation_methods_share_identical_seeded_starts" in training,
        "shared evaluation starts are not recorded",
    )
    require(
        "near_obstacle_score_count" in training
        and "mean_candidate_effective_regions" in training
        and "n_effective_regions" in training,
        "Mondrian activity diagnostics are incomplete",
    )
    require(
        "none_policy_changes_after_calibration" in training,
        "training-time coverage caveat is missing",
    )
    require(
        "abort_before_executing_infeasible_cp_action" in training
        and "--allow-infeasible-cp-diagnostic-run" in training,
        "infeasible CP actions are not strict-by-default with an explicit diagnostic override",
    )
    require(
        all(
            field in training
            for field in (
                '"xi_hat_app"',
                '"epsilon_inter"',
                '"max_u_psi"',
                '"feasibility_margin"',
                '"fallback_kind"',
                '"fallback_is_actuator_rail"',
                '"minimal_projection_verified"',
            )
        ),
        "per-active-step QP feasibility/fallback diagnostics are incomplete",
    )
    require(
        "qp_feasibility_diagnostics.jsonl" in training,
        "dedicated QP feasibility diagnostics output is missing",
    )
    require(
        "def _audit_cp_feasibility_grid" in training
        and '"discrete_grid_gate_passed"' in training
        and "The discrete pre-training CP-QP feasibility gate failed" in training,
        "the pre-training discrete Assumption-5 feasibility gate is missing",
    )
    require(
        '"primary independently trained method"' in training
        and 'if args.training_shield_method == "nominal"' in training
        and 'if args.training_shield_method == "cbvf"' in training
        and 'if args.training_shield_method == "cp"' in training
        and '"counterfactual unshielded evaluation"' in training,
        "per-run primary methods and counterfactual evaluations are not labeled distinctly",
    )

    calibration_position = training.index("train_shield = TrainingShield")
    first_training_reset = training.index(
        "obs, _ = _reset_certified_env(env, train_shield, seed=args.seed)"
    )
    first_training_loop = training.index("for step in iterator")
    require(
        calibration_position < first_training_reset < first_training_loop,
        "calibration is not completed before the first RL episode",
    )
    print("Static workflow checks: PASS")
    print(f"Finalized shield SHA-256: {shield_hash}")


if __name__ == "__main__":
    main()
