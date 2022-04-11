import gym
import sys
from os import getcwd
sys.path.append(getcwd())

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from drloco.mujoco.Ant import AntEnvV2

# Parallel environments
make_env = lambda: AntEnvV2(1)
env = make_vec_env(make_env, n_envs=4)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25000)
model.save("ppo_antv2")