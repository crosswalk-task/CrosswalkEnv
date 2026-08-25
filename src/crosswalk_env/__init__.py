from gymnasium.envs.registration import register
from crosswalk_env.environment import CrosswalkEnv

register(
    id="crosswalk-v1",
    entry_point="crosswalk_env:CrosswalkEnv",
)
