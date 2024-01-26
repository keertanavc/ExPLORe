import gymnax
import gym
from gym import spaces
import numpy as np
from gymnax.wrappers.gym import GymnaxToGymWrapper
from rlpd.wrappers.discrete2continuous import Discrete2Continuous

env, env_params = gymnax.make('DeepSea-bsuite')
env = wrap_gym(env, env_params=env_params, discrete_action=True, rescale_actions=True)

#env = GymnaxToGymWrapper(env=env, params=env_params)
#env = Discrete2Continuous(env)
X = generate_random_trajectories(env=env, num_trajectories=20, horizon=2000)

env2 = gym.make('MountainCar-v0')
AWAC_DATA_DIR = "./data"

expert_dataset = np.load(
    os.path.join(
        os.path.expanduser(AWAC_DATA_DIR), f"random_action1.npy"
    ),
    allow_pickle=True,
)

import importlib
importlib.reload(rlpd)
from rlpd.data.deepsea_datasets import DeepSeaDataset
