import gym
import liveline_gym
env = gym.make('liveline-v0', verbosity=0)

# After pasting ddpg and core.mlp_actor_critic...

from spinup.utils.run_utils import setup_logger_kwargs
logger_kwargs = setup_logger_kwargs('foo_experiment', 42)

ddpg(lambda: gym.make('liveline-v0'), actor_critic=core.mlp_actor_critic,
     ac_kwargs=dict(hidden_sizes=[256] * 2),
     gamma=0.99, seed=42, epochs=3,
     logger_kwargs=logger_kwargs)