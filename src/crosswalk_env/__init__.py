from crosswalk_env.environment import CrosswalkEnv
from gymnasium.envs.registration import register

__version__ = "0.0.1"

register(
    id="crosswalk-v1",
    entry_point="crosswalk_env:CrosswalkEnv",
)
