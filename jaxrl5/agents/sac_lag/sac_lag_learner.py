"""Implementations of algorithms for continuous control."""

from functools import partial
from typing import Dict, Optional, Sequence, Tuple

import gymnasium as gym
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training.train_state import TrainState

from jaxrl5.agents.agent import Agent
from jaxrl5.agents.sac.temperature import Temperature
from jaxrl5.agents.sac_lag.lag_multiplier import LagMultiplier
from jaxrl5.data.dataset import DatasetDict
from jaxrl5.distributions import TanhNormal
from jaxrl5.networks import MLP, Ensemble, StateActionValue, subsample_ensemble


class SACLagLearner(Agent):
    critic: TrainState
    target_critic: TrainState
    cost_critic: TrainState
    target_cost_critic: TrainState
    temp: TrainState
    lag: TrainState
    tau: float
    discount: float
    target_entropy: float
    cost_limit: float
    num_qs: int = struct.field(pytree_node=False)
    num_min_qs: Optional[int] = struct.field(
        pytree_node=False
    )  # See M in RedQ https://arxiv.org/abs/2101.05982
    backup_entropy: bool = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        lag_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        num_qs: int = 2,
        num_min_qs: Optional[int] = None,
        critic_dropout_rate: Optional[float] = None,
        critic_layer_norm: bool = False,
        target_entropy: Optional[float] = None,
        init_temperature: float = 1.0,
        init_lag: float = 0.0,
        backup_entropy: bool = True,
        cost_limit: float = 0.0,
    ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """

        action_dim = action_space.shape[-1]
        observations = observation_space.sample()
        actions = action_space.sample()

        if target_entropy is None:
            target_entropy = -action_dim / 2

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, cost_critic, temp_key = jax.random.split(rng, 5)

        actor_base_cls = partial(MLP, hidden_dims=hidden_dims, activate_final=True)
        actor_def = TanhNormal(actor_base_cls, action_dim)
        actor_params = actor_def.init(actor_key, observations)["params"]
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=actor_lr),
        )

        critic_base_cls = partial(
            MLP,
            hidden_dims=hidden_dims,
            activate_final=True,
            dropout_rate=critic_dropout_rate,
            use_layer_norm=critic_layer_norm,
        )
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_def = Ensemble(critic_cls, num=num_qs)
        critic_params = critic_def.init(critic_key, observations, actions)["params"]
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.adam(learning_rate=critic_lr),
        )
        target_critic_def = Ensemble(critic_cls, num=num_min_qs or num_qs)
        target_critic = TrainState.create(
            apply_fn=target_critic_def.apply,
            params=critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        cost_critic_params = critic_def.init(critic_key, observations, actions)[
            "params"
        ]
        cost_critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=cost_critic_params,
            tx=optax.adam(learning_rate=critic_lr),
        )
        target_critic_def = Ensemble(critic_cls, num=num_min_qs or num_qs)
        target_cost_critic = TrainState.create(
            apply_fn=target_critic_def.apply,
            params=cost_critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)["params"]
        temp = TrainState.create(
            apply_fn=temp_def.apply,
            params=temp_params,
            tx=optax.adam(learning_rate=temp_lr),
        )

        lag_multiplier_def = LagMultiplier(init_lag)
        lag_multiplier_params = lag_multiplier_def.init(temp_key)["params"]
        lag = TrainState.create(
            apply_fn=lag_multiplier_def.apply,
            params=lag_multiplier_params,
            tx=optax.adam(learning_rate=lag_lr),
        )

        return cls(
            rng=rng,
            actor=actor,
            critic=critic,
            target_critic=target_critic,
            cost_critic=cost_critic,
            target_cost_critic=target_cost_critic,
            temp=temp,
            lag=lag,
            target_entropy=target_entropy,
            tau=tau,
            discount=discount,
            num_qs=num_qs,
            num_min_qs=num_min_qs,
            backup_entropy=backup_entropy,
            cost_limit=cost_limit,
        )

    def update_actor(self, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        # L = -Q^V (s, \pi (s)) + \lambda Q^C (s, \pi (s))

        key, rng = jax.random.split(self.rng)
        key2, rng = jax.random.split(rng)
        key3, rng = jax.random.split(rng)

        def actor_loss_fn(actor_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            dist = self.actor.apply_fn({"params": actor_params}, batch["observations"])
            # see https://github.com/google-deepmind/distrax/issues/216
            actions, log_probs = dist.sample_and_log_prob(seed=key)
            qs = self.critic.apply_fn(
                {"params": self.critic.params},
                batch["observations"],
                actions,
                True,
                rngs={"dropout": key2},
            )  # training=True
            q = qs.mean(axis=0)
            
            
            qc_s = self.cost_critic.apply_fn(
                {"params": self.cost_critic.params},
                batch["observations"],
                actions,
                True,
                rngs={"dropout": key3},
            )  # training=True
            qc = qc_s.mean(axis=0)

            lmbda = self.lag.apply_fn({"params": self.lag.params})


            
            actor_loss = (
                log_probs * self.temp.apply_fn({"params": self.temp.params}) - q + lmbda * qc
            ).mean()

            return actor_loss, {"actor_loss": actor_loss, "entropy": -log_probs.mean(), "qc": qc.sum(axis=-1)}

        grads, actor_info = jax.grad(actor_loss_fn, has_aux=True)(self.actor.params)
        actor = self.actor.apply_gradients(grads=grads)

        return self.replace(actor=actor, rng=rng), actor_info

    def update_temperature(self, entropy: float) -> Tuple[Agent, Dict[str, float]]:
        def temperature_loss_fn(temp_params):
            temperature = self.temp.apply_fn({"params": temp_params})
            temp_loss = temperature * (entropy - self.target_entropy).mean()
            return temp_loss, {
                "temperature": temperature,
                "temperature_loss": temp_loss,
            }

        grads, temp_info = jax.grad(temperature_loss_fn, has_aux=True)(self.temp.params)
        temp = self.temp.apply_gradients(grads=grads)

        return self.replace(temp=temp), temp_info

    def update_lambda(self, qc: float) -> Tuple[Agent, Dict[str, float]]:
        def lambda_loss_fn(temp_params):
            lmbda = self.lag.apply_fn({"params": temp_params})
            lmbda = jnp.clip(lmbda, a_min=0.0)
            lmbda_loss = -lmbda * (qc - self.cost_limit).mean()
            return lmbda_loss, {
                "lambda": lmbda,
                "lambda_loss": lmbda_loss,
            }

        grads, lam_info = jax.grad(lambda_loss_fn, has_aux=True)(self.lag.params)
        lag = self.lag.apply_gradients(grads=grads)

        return self.replace(lag=lag), lam_info

    def update_critic(self, batch: DatasetDict) -> Tuple[TrainState, Dict[str, float]]:

        dist = self.actor.apply_fn(
            {"params": self.actor.params}, batch["next_observations"]
        )

        rng = self.rng

        key, rng = jax.random.split(rng)
        next_actions, next_log_probs = dist.sample_and_log_prob(seed=key)

        # Used only for REDQ.
        key, rng = jax.random.split(rng)
        target_params = subsample_ensemble(
            key, self.target_critic.params, self.num_min_qs, self.num_qs
        )

        key, rng = jax.random.split(rng)
        next_qs = self.target_critic.apply_fn(
            {"params": target_params},
            batch["next_observations"],
            next_actions,
            True,
            rngs={"dropout": key},
        )  # training=True
        next_q = next_qs.min(axis=0)

        target_q = batch["rewards"] + self.discount * batch["masks"] * next_q

        if self.backup_entropy:
            target_q -= (
                self.discount
                * batch["masks"]
                * self.temp.apply_fn({"params": self.temp.params})
                * next_log_probs
            )

        key, rng = jax.random.split(rng)

        def critic_loss_fn(critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qs = self.critic.apply_fn(
                {"params": critic_params},
                batch["observations"],
                batch["actions"],
                True,
                rngs={"dropout": key},
            )  # training=True
            critic_loss = ((qs - target_q) ** 2).mean()
            return critic_loss, {"critic_loss": critic_loss, "q": qs.mean()}

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(self.critic.params)
        critic = self.critic.apply_gradients(grads=grads)

        target_critic_params = optax.incremental_update(
            critic.params, self.target_critic.params, self.tau
        )
        target_critic = self.target_critic.replace(params=target_critic_params)

        return self.replace(critic=critic, target_critic=target_critic, rng=rng), info

    def update_cost_critic(
        self, batch: DatasetDict
    ) -> Tuple[TrainState, Dict[str, float]]:

        dist = self.actor.apply_fn(
            {"params": self.actor.params}, batch["next_observations"]
        )

        rng = self.rng

        key, rng = jax.random.split(rng)
        next_actions, next_log_probs = dist.sample_and_log_prob(seed=key)

        # Used only for REDQ.
        key, rng = jax.random.split(rng)
        target_params = subsample_ensemble(
            key, self.target_cost_critic.params, self.num_min_qs, self.num_qs
        )

        key, rng = jax.random.split(rng)
        next_qs = self.target_cost_critic.apply_fn(
            {"params": target_params},
            batch["next_observations"],
            next_actions,
            True,
            rngs={"dropout": key},
        )  # training=True
        next_q = next_qs.min(axis=0)

        target_q = batch["costs"] + self.discount * batch["masks"] * next_q

        key, rng = jax.random.split(rng)

        def critic_loss_fn(critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qs = self.cost_critic.apply_fn(
                {"params": critic_params},
                batch["observations"],
                batch["actions"],
                True,
                rngs={"dropout": key},
            )  # training=True
            cost_critic_loss = ((qs - target_q) ** 2).mean()
            return cost_critic_loss, {"cost_critic_loss": cost_critic_loss, "qc": qs.mean()}

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(self.cost_critic.params)
        cost_critic = self.cost_critic.apply_gradients(grads=grads)

        target_cost_critic_params = optax.incremental_update(
            cost_critic.params, self.target_cost_critic.params, self.tau
        )
        target_cost_critic = self.target_cost_critic.replace(
            params=target_cost_critic_params
        )

        return (
            self.replace(
                cost_critic=cost_critic, target_cost_critic=target_cost_critic, rng=rng
            ),
            info,
        )

    @partial(jax.jit, static_argnames="utd_ratio")
    def update(self, batch: DatasetDict, utd_ratio: int):

        new_agent = self
        for i in range(utd_ratio):

            def slice(x):
                assert x.shape[0] % utd_ratio == 0
                batch_size = x.shape[0] // utd_ratio
                return x[batch_size * i : batch_size * (i + 1)]

            mini_batch = jax.tree_util.tree_map(slice, batch)
            new_agent, critic_info = new_agent.update_critic(mini_batch)
            new_agent, cost_critic_info = new_agent.update_cost_critic(mini_batch)

        new_agent, actor_info = new_agent.update_actor(mini_batch)
        new_agent, temp_info = new_agent.update_temperature(actor_info["entropy"])
        new_agent, lambda_info = new_agent.update_lambda(actor_info["qc"])

        return new_agent, {**actor_info, **critic_info, **temp_info, **cost_critic_info, **lambda_info}
