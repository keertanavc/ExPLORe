import os
import sys
import pickle
from functools import partial
import jax
import gym
from gym import spaces
import gymnax
from gymnax.environments import spaces # not exactly the same as gym.spaces
import numpy as np
import tqdm
from absl import app, flags
from flax.core import frozen_dict
from absl import logging
import jax.numpy as jnp
logging.set_verbosity(logging.FATAL)
from flax.training import checkpoints
from ml_collections import config_flags
from rlpd.agents import SACLearner, RM, RND, BCAgent
from rlpd.data import ReplayBuffer, Dataset
from rlpd.wrappers import wrap_gym
from rlpd.evaluation import evaluate
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import plotly.express as px
import wandb
import glob
import os
import time
from datetime import datetime

#from rlpd.data.d4rl_datasets import D4RLDataset, filter_antmaze
#from rlpd.data.binary_datasets import BinaryDataset
from rlpd.data.deepsea_datasets import DeepSeaDataset


FLAGS = flags.FLAGS

flags.DEFINE_string("project_name", "deepsea_explore", "wandb project name.")
flags.DEFINE_string("env_name", "DeepSea-bsuite", "Deep sea behavior suite")
flags.DEFINE_float("offline_ratio", 0.5, "Offline ratio.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 100, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 10000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", 2000, "Number of training steps.") #2000 for deep sea
flags.DEFINE_boolean("discrete_action", True, "Whether env actions are discrete") #True for deep sea`
flags.DEFINE_boolean("is_deterministic", True, "whether environment is deterministic") #True for deep sea`
flags.DEFINE_integer(
    "start_training", 5000, "Number of training steps to start training."
) #might need to change this also later

flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("checkpoint_model", False, "Save agent checkpoint on evaluation.")
flags.DEFINE_boolean(
    "checkpoint_buffer", False, "Save agent replay buffer on evaluation."
)
# flags.DEFINE_boolean(
#     "binary_include_bc", True, "Whether to include BC data in the binary datasets."
# ) #not relevant I guess

flags.DEFINE_integer("utd_ratio", 20, "Update to data ratio.")
flags.DEFINE_string("offline_relabel_type", "pred", "one of [gt/pred/min]") #=pred for explore
flags.DEFINE_string("exp_prefix", "exp_data/default", "log directory")
flags.DEFINE_boolean("use_rnd_offline", True, "Whether to use rnd offline.") #optimistic rewards for offline data
flags.DEFINE_boolean("use_rnd_online", False, "Whether to use rnd online.") #no optimism for online data
flags.DEFINE_float("bc_pretrain_rollin", 0.0, "rollin coeff")

flags.DEFINE_integer(
    "bc_pretrain_steps",
    5000,
    "Pre-train BC policy for a number of steps on pure offline data",
)

flags.DEFINE_integer("reset_rm_every", -1, "Reset the reward network every N env steps") # not resetting anything
flags.DEFINE_string("filter_data_mode", "all", "Strategy to filter offline data") # not sure what this is!
FLAGS(sys.argv)

# not changing the parameters for any newtworks for now
config_flags.DEFINE_config_file(
    "config",
    "configs/rlpd_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    "rm_config",
    "configs/rm_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    "rnd_config",
    "configs/rnd_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

# helper function for combining dictionaries
def combine(one_dict, other_dict):
    combined = {}
    for k, v in one_dict.items():
        if isinstance(v, dict):
            combined[k] = combine(v, other_dict[k])
        else:
            tmp = np.empty(
                (v.shape[0] + other_dict[k].shape[0], *v.shape[1:]), dtype=v.dtype
            )
            tmp[0::2] = v
            tmp[1::2] = other_dict[k]
            combined[k] = tmp
    return combined

# add prefix to dictionary keys
def add_prefix(prefix, dict):
    return {prefix + k: v for k, v in dict.items()}

@partial(jax.jit, static_argnames=("R",))
def check_overlap(coord, observations, R):
    return jnp.any(jnp.all(jnp.abs(coord - observations[..., :2]) <= R, axis=-1))


# def view_data_distribution(viz_env, ds):
#     vobs = ds.dataset_dict["observations"][..., :2]
#     return plot_points(viz_env, vobs[:, 0], vobs[:, 1])
env, env_params = gymnax.make(FLAGS.env_name)
env = wrap_gym(env, rescale_actions=True)
env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
eval_env, eval_env_params = gymnax.make(FLAGS.env_name)
eval_env = wrap_gym(eval_env, FLAGS.discrete_action, rescale_actions=True)
ds = DeepSeaDataset(env)

action_space = env.action_space
kwargs = dict(FLAGS.config)
model_cls = kwargs.pop("model_cls")
agent = globals()[model_cls].create(
    FLAGS.seed, env.observation_space, action_space, **kwargs
)
kwargs = dict(FLAGS.rnd_config)
model_cls = kwargs.pop("model_cls")
rnd = globals()[model_cls].create(
    FLAGS.seed + 123, env.observation_space, action_space, **kwargs
)
kwargs = dict(FLAGS.rm_config)
model_cls = kwargs.pop("model_cls")
rm = globals()[model_cls].create(
    FLAGS.seed + 123, env.observation_space, action_space, **kwargs
)
# Pre-training
record_step = 0
observation, done = env.reset(), False
online_trajs = []
online_traj = [observation]
rng = jax.random.PRNGKey(seed=FLAGS.seed)
rollin_enabled = False
env_step = 0
action = env.action_space.sample()
next_observation, reward, done, terminated, info = env.step(action)
env_step += 1
online_traj.append(next_observation)
mask = 1.0
