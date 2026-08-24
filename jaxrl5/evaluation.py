from typing import Dict

import gymnasium as gym
import numpy as np

from redexp.wrapper.record_costs import RecordEpisodeStatistics


def evaluate(
    agent,
    env: gym.Env,
    num_episodes: int,
    seed: int,
) -> Dict[str, float]:
    env = RecordEpisodeStatistics(env, deque_size=num_episodes)

    for i in range(num_episodes):
        observation, _ = env.reset(seed=seed)
        done = False
        while not done:
            action, agent = agent.eval_actions(observation)
            observation, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    return {"return": np.mean(env.return_queue), "cost:": np.mean(env.cost_queue)}
