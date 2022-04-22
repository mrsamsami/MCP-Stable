import gym
from tqdm import tqdm

env = gym.make('Ant-v2')
from copy import copy
observation = env.reset()
initial_vals_set = False
import numpy as np

for _ in tqdm(range(100000)):
    action = env.action_space.sample() # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)
    if not initial_vals_set:
        the_min = copy(observation)
        the_max = copy(observation)
        initial_vals_set = True
    else:
        the_min = np.minimum(the_min, observation)
        the_max = np.maximum(the_max, observation)

    if done:
        observation = env.reset()

print("min:")
print(the_min)

print("\n\nmax:")
print(the_max)
env.close()

import numpy as np
dir_path=""
direction = 3
qpos = np.random.uniform(size=(2, 300), low=-0.1, high=0.1)
old_file_path = 'data/ppo_ant_direction{}.npy'.format(direction)
new_file_path = 'data/new_ppo_ant_direction{}.npy'.format(direction)
data = np.load(dir_path + old_file_path)
new_data = np.concatenate((qpos, data), 0)
np.save(new_file_path, new_data)