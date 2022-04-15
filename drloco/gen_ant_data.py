import numpy as np

import drloco
import gym
import sys
from drloco.common.utils import save_as_gif
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

direction = int(sys.argv[1])
time_steps = int(sys.argv[2]) if len(sys.argv) == 3 else 100
env = make_vec_env('NewAnt-v2', n_envs=1, env_kwargs={'direction': direction})
model = PPO.load("models/ppo_ant_direction{}".format(direction))

obs = env.reset()
data = []

for t in range(time_steps):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)

    if dones[0]:
        break

    data.append(obs[:, :27])

env.close()
data = np.concatenate(data, axis = 0).T
np.save("data/ppo_ant_direction{}".format(direction), data)