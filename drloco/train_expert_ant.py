import drloco
import gym
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

direction = int(sys.argv[1])
total_timesteps = int(sys.argv[2]) if len(sys.argv) == 3 else 2000000
env = make_vec_env('NewAnt-v2', n_envs=4, env_kwargs={'direction': direction})
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=total_timesteps)
model.save("models/ppo_ant_direction{}".format(direction))