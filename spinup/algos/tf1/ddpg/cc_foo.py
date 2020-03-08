import gym
import liveline_gym

from spinup.algos.tf1.ddpg.ddpg import ddpg
from spinup.algos.tf1.ddpg import core

from spinup.utils.run_utils import setup_logger_kwargs
logger_kwargs = setup_logger_kwargs('foo_experiment', 42)

ddpg(lambda: gym.make('liveline-v0'), actor_critic=core.mlp_actor_critic,
     ac_kwargs=dict(hidden_sizes=[256] * 2),
     gamma=0.99, seed=42,

     # DEBUG: Delete later
     steps_per_epoch=40, epochs=1, start_steps=100, update_after=10, update_every=5,
     max_ep_len=10, num_test_episodes=1,

     logger_kwargs=logger_kwargs)
