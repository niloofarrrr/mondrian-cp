import flax.linen as nn
import jax.numpy as jnp


class LagMultiplier(nn.Module):
    initial_lag: float = 0.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        lag = self.param(
            "lag",
            init_fn=lambda key: jnp.full((), self.initial_lag),
        )
        return lag
