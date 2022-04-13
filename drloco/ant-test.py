import drloco
import gym
import sys
from os import getcwd
# sys.path.append(getcwd())

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from drloco.mujoco.ant import AntEnvV2

# Parallel environments
dir = int(sys.argv[1])
env = make_vec_env('NewAnt-v2', n_envs=4, env_kwargs={'direction': dir})
# env = make_vec_env('Ant-v2', n_envs=4)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=2000000)
model.save("ppo_ant{}".format(dir))