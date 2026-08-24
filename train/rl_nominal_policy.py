"""
Policy loader used by conformal_shield.py.

RLNominalPolicy loads a trained SAC or SAC-Lagrangian actor from train/train_sac_lag.py
and exposes the same call interface as the old hand-written nominal controller:

    u_nom = policy(state)

The conformal shield then uses u_nom exactly as before.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np


def wrap_angle(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class HeuristicNominalPolicy:
    """Optional fallback for debugging only; the paper experiments should use RLNominalPolicy."""

    def __init__(self, world, control_bounds, k_heading: float = 3.0):
        self.world = world
        self.control_bounds = control_bounds
        self.k_heading = float(k_heading)

    def __call__(self, s: np.ndarray) -> float:
        x, y, th = s
        gx, gy = self.world.goal
        desired = math.atan2(gy - y, gx - x)
        u = self.k_heading * wrap_angle(float(desired - th))
        return float(np.clip(u, self.control_bounds.lo, self.control_bounds.hi))

    def describe(self) -> str:
        return "heuristic proportional heading controller"


def _latest_checkpoint_step(checkpoint_dir: Path) -> int:
    candidates = list(checkpoint_dir.glob("params_*.pkl"))
    if not candidates:
        raise FileNotFoundError(
            f"No params_*.pkl checkpoint found in {checkpoint_dir}. "
            "Run train/train_sac_lag.py first."
        )
    steps = []
    for path in candidates:
        match = re.match(r"params_(\d+)\.pkl$", path.name)
        if match:
            steps.append(int(match.group(1)))
    if not steps:
        raise FileNotFoundError(f"No numeric params_<step>.pkl checkpoint found in {checkpoint_dir}.")
    return max(steps)


def _read_training_config(checkpoint_dir: Path) -> dict:
    config_path = checkpoint_dir / "policy_config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


class RLNominalPolicy:
    """Loaded SAC or SAC-Lag actor used as the nominal controller."""

    def __init__(
        self,
        world,
        control_bounds,
        checkpoint_dir: Union[str, Path] = "rl_checkpoints/reach_avoid_nominal_good",
        checkpoint_step: Union[int, str] = "latest",
        seed: int = 42,
        hidden_dims: Sequence[int] = (256, 256),
        deterministic: bool = True,
    ):
        self.world = world
        self.control_bounds = control_bounds
        self.checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
        self.deterministic = bool(deterministic)
        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(
                f"RL checkpoint directory does not exist: {self.checkpoint_dir}. "
                "Train first with train/train_sac_lag.py."
            )

        cfg = _read_training_config(self.checkpoint_dir)
        self.hidden_dims: Tuple[int, ...] = tuple(cfg.get("hidden_dims", list(hidden_dims)))
        self.seed = int(cfg.get("seed", seed))
        self.algorithm = str(cfg.get("algorithm", "sac_lag" if "sac_lag" in str(self.checkpoint_dir) else "sac"))
        self.cost_limit = float(cfg.get("cost_limit", 999.0 if self.algorithm == "sac" else 0.1))
        self.init_temperature = float(cfg.get("init_temperature", 0.1))

        if str(checkpoint_step).lower() in {"latest", "-1"}:
            self.checkpoint_step = _latest_checkpoint_step(self.checkpoint_dir)
        else:
            self.checkpoint_step = int(checkpoint_step)

        try:
            import gymnasium as gym
            from jaxrl5.agents import SACLearner, SACLagLearner
            from jaxrl5.agents.agent import restore_agent
        except Exception as exc:
            raise ImportError(
                "RLNominalPolicy requires the RL dependencies used by jaxrl5 "
                "(jax, flax, optax, gymnasium, etc.). Install the environment from "
                "the base repository before running the RL nominal controller."
            ) from exc

        obs_space = gym.spaces.Box(
            low=np.array([world.xmin, world.ymin, -1.0, -1.0, world.xmin, world.ymin], dtype=np.float32),
            high=np.array([world.xmax, world.ymax, 1.0, 1.0, world.xmax, world.ymax], dtype=np.float32),
            dtype=np.float32,
        )
        action_space = gym.spaces.Box(
            low=np.array([control_bounds.lo], dtype=np.float32),
            high=np.array([control_bounds.hi], dtype=np.float32),
            dtype=np.float32,
        )
        if self.algorithm == "sac":
            agent = SACLearner.create(
                self.seed,
                obs_space,
                action_space,
                hidden_dims=self.hidden_dims,
                init_temperature=self.init_temperature,
            )
        elif self.algorithm == "sac_lag":
            agent = SACLagLearner.create(
                self.seed,
                obs_space,
                action_space,
                hidden_dims=self.hidden_dims,
                init_temperature=self.init_temperature,
                cost_limit=self.cost_limit,
            )
        else:
            raise ValueError(f"Unsupported saved RL algorithm '{self.algorithm}' in policy_config.json")
        self.agent = restore_agent(agent, str(self.checkpoint_dir), self.checkpoint_step)

    def _obs(self, s: np.ndarray) -> np.ndarray:
        gx, gy = self.world.goal
        th = float(s[2])
        return np.array([s[0], s[1], np.cos(th), np.sin(th), gx, gy], dtype=np.float32)

    def __call__(self, s: np.ndarray) -> float:
        obs = self._obs(np.asarray(s, dtype=np.float32))
        if self.deterministic:
            action, self.agent = self.agent.eval_actions(obs)
        else:
            action, self.agent = self.agent.sample_actions(obs)
        u = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        return float(np.clip(u, self.control_bounds.lo, self.control_bounds.hi))

    def describe(self) -> str:
        return f"{self.algorithm} RL checkpoint {self.checkpoint_dir}/params_{self.checkpoint_step}.pkl"
