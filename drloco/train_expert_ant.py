import drloco
import gym
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

direction = int(sys.argv[1])
total_timesteps = int(sys.argv[2]) if len(sys.argv) >= 3 else 2000000
load = bool(sys.argv[3]) if len(sys.argv) == 4 else False
env = make_vec_env('NewAnt-v2', n_envs=4, env_kwargs={'direction': direction})
if not load:
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="logs/ppo_ant_direction{}".format(direction))
else:
    model = PPO.load("models/ppo_ant_direction{}".format(direction), tensorboard_log="logs/ppo_ant_direction{}".format(direction))
    model.set_env(env)

model.learn(total_timesteps=total_timesteps)
model.save("models/ppo_ant_direction{}".format(direction))