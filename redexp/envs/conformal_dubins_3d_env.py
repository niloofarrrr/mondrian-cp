"""
Gymnasium simulation environment for training the RL nominal controller used by
conformal_shield.py.

This file is training-only.  It contains the Dubins simulation dynamics and the
reward/cost used by SAC-Lag.  The online CBVF/CP shield is not implemented here;
train/train_sac_lag.py optionally wraps actions with the CP shield before they
are sent to this environment.

Training revision:
  * The task reward is kept simple: r(x, u) = -distance_to_goal.
  * Safety is handled through the CMDP cost plus an optional hard terminal
    collision penalty used only for training stabilization.
  * The default safety cost is a signed-distance shaped cost
        c(x,u) = max(0, threshold - l(x)),
    where l(x) is the signed distance to the obstacle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np


def wrap_angle(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def rk4_step(f, x: np.ndarray, dt: float) -> np.ndarray:
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    x_next = np.asarray(x_next, dtype=np.float32)
    x_next[2] = wrap_angle(float(x_next[2]))
    return x_next


@dataclass(frozen=True)
class ConformalDubinsConfig:
    xmin: float = -4.0
    xmax: float = 4.0
    ymin: float = -4.0
    ymax: float = 4.0
    goal: Tuple[float, float] = (2.0, 2.0)
    goal_radius: float = 0.30
    obstacle_center: Tuple[float, float] = (0.0, 0.0)
    obstacle_radius: float = 0.50
    robot_radius: float = 0.20
    speed_min: float = 0.0
    speed: float = 0.60
    beta_u: float = 0.50
    u_min: float = -1.0
    u_max: float = 1.0
    dt: float = 0.05
    horizon: int = 400
    initial_state: Tuple[float, float, float] = (-2.0, -2.0, math.pi / 4.0)
    random_start: bool = True
    start_jitter_xy: float = 0.60
    start_jitter_theta: float = 0.35

    # Keep the reward equal to -distance_to_goal by default.  The following
    # optional reward terms are retained only for ablations/debugging.
    goal_bonus: float = 0.0
    action_penalty: float = 0.0
    unsafe_penalty: float = 0.0
    out_of_bounds_penalty: float = 250.0
    collision_terminal_penalty: float = 0.0

    # Safety cost used by SAC-Lag.  With signed-distance l(x), the default is
    # c(x,u) = max(0, safety_cost_margin - l(x)).
    dense_safety_cost: bool = True
    dense_safety_weight: float = 0.3
    safety_cost_margin: float = 0.10
    binary_unsafe_cost: bool = False
    terminate_on_unsafe: bool = False


class ConformalDubins3dEnv(gym.Env):
    """
    Dubins-car RL training environment matched to conformal_shield.py.

    State: [x, y, theta]
    Action: [normalized throttle/brake, normalized yaw] in [-1, 1]^2
    Observation: [x, y, cos(theta), sin(theta), goal_x, goal_y]

    The default reward is exactly -distance_to_goal.  The default cost is a
    continuous near-obstacle cost based on signed distance.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        speed: float = 0.60,
        speed_min: float = 0.0,
        beta_u: float = 0.50,
        dt: float = 0.05,
        horizon: int = 400,
        initial_state: Tuple[float, float, float] = (-2.0, -2.0, math.pi / 4.0),
        random_start: bool = True,
        start_jitter_xy: float = 0.60,
        start_jitter_theta: float = 0.35,
        dense_safety_cost: bool = True,
        dense_safety_weight: float = 0.3,
        safety_cost_margin: float = 0.10,
        binary_unsafe_cost: bool = False,
        goal_bonus: float = 0.0,
        action_penalty: float = 0.0,
        unsafe_penalty: float = 0.0,
        out_of_bounds_penalty: float = 250.0,
        collision_terminal_penalty: float = 0.0,
        terminate_on_unsafe: bool = False,
    ):
        initial_state_tuple = tuple(float(value) for value in initial_state)
        if len(initial_state_tuple) != 3 or not np.isfinite(initial_state_tuple).all():
            raise ValueError("initial_state must contain three finite values.")
        self.cfg = ConformalDubinsConfig(
            speed=float(speed),
            speed_min=float(speed_min),
            beta_u=float(beta_u),
            dt=float(dt),
            horizon=int(horizon),
            initial_state=initial_state_tuple,
            random_start=bool(random_start),
            start_jitter_xy=float(start_jitter_xy),
            start_jitter_theta=float(start_jitter_theta),
            dense_safety_cost=bool(dense_safety_cost),
            dense_safety_weight=float(dense_safety_weight),
            safety_cost_margin=float(safety_cost_margin),
            binary_unsafe_cost=bool(binary_unsafe_cost),
            goal_bonus=float(goal_bonus),
            action_penalty=float(action_penalty),
            unsafe_penalty=float(unsafe_penalty),
            out_of_bounds_penalty=float(out_of_bounds_penalty),
            collision_terminal_penalty=float(collision_terminal_penalty),
            terminate_on_unsafe=bool(terminate_on_unsafe),
        )
        self.render_mode = render_mode
        self.state = np.asarray(self.cfg.initial_state, dtype=np.float32)
        self.t = 0

        self.action_space = gym.spaces.Box(
            low=np.array([self.cfg.u_min, self.cfg.u_min], dtype=np.float32),
            high=np.array([self.cfg.u_max, self.cfg.u_max], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=np.array([self.cfg.xmin, self.cfg.ymin, -1.0, -1.0, self.cfg.xmin, self.cfg.ymin], dtype=np.float32),
            high=np.array([self.cfg.xmax, self.cfg.ymax, 1.0, 1.0, self.cfg.xmax, self.cfg.ymax], dtype=np.float32),
            dtype=np.float32,
        )

    def sample_initial_state(self, rng) -> np.ndarray:
        """Sample from the one shared reset distribution used by RL and calibration."""
        base = np.asarray(self.cfg.initial_state, dtype=np.float32)
        if self.cfg.random_start:
            for _ in range(10_000):
                candidate = base.copy()
                candidate[0] += rng.uniform(-self.cfg.start_jitter_xy, self.cfg.start_jitter_xy)
                candidate[1] += rng.uniform(-self.cfg.start_jitter_xy, self.cfg.start_jitter_xy)
                candidate[2] = wrap_angle(float(candidate[2] + rng.uniform(-self.cfg.start_jitter_theta, self.cfg.start_jitter_theta)))
                if self.is_safe_state(candidate):
                    return candidate.astype(np.float32)
            else:
                raise RuntimeError("Could not sample a safe initial state for RL training.")
        return base.copy()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        self.state = self.sample_initial_state(self.np_random)
        self.t = 0
        return self._get_obs(), {"cost": 0.0, "safe_margin": self.safety_margin_value(self.state)}

    def physical_controls(self, action: np.ndarray) -> Tuple[float, float]:
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(2), self.cfg.u_min, self.cfg.u_max)
        throttle, yaw = float(action[0]), float(action[1])
        speed = self.cfg.speed_min + 0.5 * (throttle + 1.0) * (self.cfg.speed - self.cfg.speed_min)
        return float(speed), float(self.cfg.beta_u * yaw)

    def dynamics(self, s: np.ndarray, action: np.ndarray) -> np.ndarray:
        th = float(s[2])
        speed, yaw_rate = self.physical_controls(action)
        return np.array([
            speed * np.cos(th),
            speed * np.sin(th),
            yaw_rate,
        ], dtype=np.float32)

    def step(self, action):
        u = np.clip(np.asarray(action, dtype=np.float32).reshape(2), self.cfg.u_min, self.cfg.u_max)
        speed, yaw_rate = self.physical_controls(u)
        self.state = rk4_step(lambda z: self.dynamics(z, u), self.state, self.cfg.dt)
        self.t += 1

        dist_to_goal = self.goal_distance(self.state)
        margin = self.safety_margin_value(self.state)
        unsafe = margin < 0.0
        goal = self.goal_reached(self.state)
        out_of_bounds = not (self.cfg.xmin <= self.state[0] <= self.cfg.xmax and self.cfg.ymin <= self.state[1] <= self.cfg.ymax)

        reward = -dist_to_goal
        if self.cfg.action_penalty > 0.0:
            reward -= self.cfg.action_penalty * float(np.dot(u, u))
        if goal and self.cfg.goal_bonus != 0.0:
            reward += self.cfg.goal_bonus
        if unsafe and self.cfg.unsafe_penalty != 0.0:
            reward -= self.cfg.unsafe_penalty
        if unsafe and self.cfg.collision_terminal_penalty != 0.0:
            reward -= self.cfg.collision_terminal_penalty
        if out_of_bounds:
            reward -= self.cfg.out_of_bounds_penalty

        terminated = bool(goal or out_of_bounds or (self.cfg.terminate_on_unsafe and unsafe))
        truncated = bool(self.t >= self.cfg.horizon)

        cost, cost_info = self.safety_cost_from_margin(margin, out_of_bounds=out_of_bounds)
        shaped_cost = cost_info["shaped_cost"]
        binary_cost = cost_info["binary_cost"]

        info = {
            "cost": cost,
            "shaped_cost": float(shaped_cost),
            "binary_cost": float(binary_cost),
            "unsafe": bool(unsafe),
            "reach_goal": bool(goal),
            "safe_margin": float(margin),
            "goal_distance": float(dist_to_goal),
            "applied_action": [float(value) for value in u],
            "applied_speed": float(speed),
            "applied_yaw_rate": float(yaw_rate),
            "collision_terminal_penalty": float(self.cfg.collision_terminal_penalty if unsafe else 0.0),
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        gx, gy = self.cfg.goal
        th = float(self.state[2])
        return np.array([self.state[0], self.state[1], np.cos(th), np.sin(th), gx, gy], dtype=np.float32)

    def safety_margin_value(self, s: np.ndarray) -> float:
        """Signed distance to the obstacle boundary."""
        cx, cy = self.cfg.obstacle_center
        r = self.cfg.obstacle_radius + self.cfg.robot_radius
        dx = float(s[0]) - cx
        dy = float(s[1]) - cy
        return float(np.sqrt(dx ** 2 + dy ** 2) - r)

    def safety_cost_from_margin(self, margin: float, out_of_bounds: bool = False) -> Tuple[float, Dict[str, float]]:
        """Training safety cost from signed distance.

        This is separated from ``step`` so the training script can compute a
        counterfactual cost for the raw nominal action before any optional
        shield correction. The reward still uses the actual environment step.
        """
        unsafe = margin < 0.0
        shaped_cost = 0.0
        if self.cfg.dense_safety_cost:
            shaped_cost = self.cfg.dense_safety_weight * max(0.0, self.cfg.safety_cost_margin - float(margin))
        binary_cost = 1.0 if (self.cfg.binary_unsafe_cost and (unsafe or out_of_bounds)) else 0.0
        oob_cost = 1.0 if out_of_bounds else 0.0
        cost = float(shaped_cost + binary_cost + oob_cost)
        return cost, {
            "shaped_cost": float(shaped_cost),
            "binary_cost": float(binary_cost),
            "oob_cost": float(oob_cost),
            "unsafe": float(unsafe),
            "out_of_bounds": float(out_of_bounds),
            "safe_margin": float(margin),
        }

    def cost_for_state_action(self, state: np.ndarray, action: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """One-step counterfactual safety cost for a raw nominal action.

        Used by ``--cost-on-nominal-action`` to avoid the failure mode where a
        shielded action masks the cost that the nominal policy would have
        incurred. This method has no side effects on the environment state.
        """
        u = np.clip(np.asarray(action, dtype=np.float32).reshape(2), self.cfg.u_min, self.cfg.u_max)
        next_state = rk4_step(lambda z: self.dynamics(z, u), np.asarray(state, dtype=np.float32), self.cfg.dt)
        margin = self.safety_margin_value(next_state)
        out_of_bounds = not (self.cfg.xmin <= next_state[0] <= self.cfg.xmax and self.cfg.ymin <= next_state[1] <= self.cfg.ymax)
        cost, info = self.safety_cost_from_margin(margin, out_of_bounds=out_of_bounds)
        info.update({
            "action": [float(value) for value in u],
            "next_state_x": float(next_state[0]),
            "next_state_y": float(next_state[1]),
            "next_state_theta": float(next_state[2]),
        })
        return cost, info

    def is_safe_state(self, s: np.ndarray) -> bool:
        return self.safety_margin_value(s) >= 0.0

    def goal_distance(self, s: np.ndarray) -> float:
        gx, gy = self.cfg.goal
        return float(np.linalg.norm(np.asarray(s[:2], dtype=np.float32) - np.array([gx, gy], dtype=np.float32)))

    def goal_reached(self, s: np.ndarray) -> bool:
        return self.goal_distance(s) <= self.cfg.goal_radius + self.cfg.robot_radius

    def render(self):
        if self.render_mode == "rgb_array":
            return np.zeros((64, 64, 3), dtype=np.uint8)
        return None
