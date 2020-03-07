import gym
# import liveline_gym
# env = gym.make('liveline-v0', verbosity=0)

from spinup.algos.tf1.ddpg.ddpg import ReplayBuffer, ddpg
# from spinup.algos.tf1.ddpg.core import mlp_actor_critic, mlp, placeholder, placeholders, get_vars, count_vars
from spinup.algos.tf1.ddpg import core

from spinup.utils.run_utils import setup_logger_kwargs
logger_kwargs = setup_logger_kwargs('foo_experiment', 42)

ddpg(lambda: gym.make('liveline-v0'), actor_critic=core.mlp_actor_critic,
     ac_kwargs=dict(hidden_sizes=[256] * 2),
     gamma=0.99, seed=42, epochs=3,
     logger_kwargs=logger_kwargs)