from chex._src.pytypes import PRNGKey
from gymnax.environments.minatar.breakout import MinBreakout
from gymnax.environments import EnvParams
from .env import Environment


class Breakout(MinBreakout, Environment):
    def init_env(self, key: PRNGKey, params: EnvParams) -> EnvParams:
        return params  # TODO: fix this
