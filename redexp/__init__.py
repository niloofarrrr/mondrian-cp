from gymnasium.envs.registration import register

register(
    id="ConformalDubins3d-v0",
    entry_point="redexp.envs:ConformalDubins3dEnv",
)
