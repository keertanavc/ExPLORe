import gymnax
import gym
import numpy as np
import jax
import chex
import jax.numpy as jnp
from typing import Callable
from gym import spaces
from gymnax.wrappers.gym import GymnaxToGymWrapper
from gym.wrappers.flatten_observation import FlattenObservation

def convert2continuous(action):
    if action == 0:
        return np.array([1 ,  -1])
    elif action == 1:
        return np.array([-1 ,  1])
    else:
        raise NotImplementedError

def generate_random_trajectories(env, num_trajectories, horizon):
    all_trajectories = []
    keys = ['observations', 'actions', 'rewards', 'next_observations', 'terminals', 'agent_infos', 'env_infos']
    for i in range(num_trajectories):
        trajectory = {}
        for key in keys:
            trajectory[key] = []
        obs, state = env.reset()
        done = False
        for i in range(horizon):
            if done:
                break
            action = env.action_space.sample()
            next_obs, next_state, reward, done, info = env.step(action)
            trajectory['observations'].append({'observation': obs, 'state_observation': state})
            trajectory['actions'].append(convert2continuous(action))
            trajectory['rewards'].append(reward)
            trajectory['next_observations'].append({'observation': next_obs, 'state_observation': next_state})
            trajectory['terminals'].append(done)
            trajectory['agent_infos'].append({})
            trajectory['env_infos'].append({})
        trajectory['rewards'] = np.array(trajectory['rewards'])
        trajectory['terminals'] = np.array(trajectory['terminals'])
        all_trajectories.append(trajectory)
    return all_trajectories

env, env_params = gymnax.make('DeepSea-bsuite')
env = GymnaxToGymWrapper(env=env, params=env_params)
env = FlattenObservation(env)
X = generate_random_trajectories(env=env, num_trajectories=25, horizon=2000)
with open('./data/random_action1.npy', 'wb') as f:
    np.save(f, X)
