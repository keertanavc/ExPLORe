import gym
import numpy as np
from gym.spaces import Box, Dict, Discrete


class Discrete2Continuous(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.n =  env.action_space.n
        self.action_space = Box(low=-1.0, high=1.0, shape=(self.n, ), dtype=np.float32)
    def action(self, act):
        prob = np.exp(act * 10)/np.sum(np.exp(act * 10)) # multiplying by 10 to make sure a large range of policies can be used
        action = np.random.choice(self.n, 1, p=prob)
        return action[0]
