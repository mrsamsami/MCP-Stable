# import gym
# from tqdm import tqdm

# env = gym.make('Ant-v2')
# from copy import copy
# observation = env.reset()
# initial_vals_set = False
# import numpy as np

# for _ in tqdm(range(100000)):
#     action = env.action_space.sample() # your agent here (this takes random actions)
#     observation, reward, done, info = env.step(action)
#     if not initial_vals_set:
#         the_min = copy(observation)
#         the_max = copy(observation)
#         initial_vals_set = True
#     else:
#         the_min = np.minimum(the_min, observation)
#         the_max = np.maximum(the_max, observation)

#     if done:
#         observation = env.reset()

# print("min:")
# print(the_min)

# print("\n\nmax:")
# print(the_max)
# env.close()

# import numpy as np
# dir_path=""
# direction = 3
# qpos = np.random.uniform(size=(2, 300), low=-0.1, high=0.1)
# old_file_path = 'data/ppo_ant_direction{}.npy'.format(direction)
# new_file_path = 'data/new_ppo_ant_direction{}.npy'.format(direction)
# data = np.load(dir_path + old_file_path)
# new_data = np.concatenate((qpos, data), 0)
# np.save(new_file_path, new_data)

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from drloco.common.utils import lowpass_filter_data, smooth_exponential
import seaborn as sns

labels = [
    "TORSO_X", "TORSO_Y", "TORSO_Z",
    "TORSO_ANGX", "TORSO_ANGY", "TORSO_ANGZ", "TORSO_ANGW",
"HIP1_ANG", "ANKLE1_ANG", "HIP2_ANG", "ANKLE2_ANG",
"HIP3_ANG", "ANKLE3_ANG", "HIP4_ANG", "ANKLE5_ANG",
"TORSO_VX", "TORSO_VY", "TORSO_VZ",
"TORSO_ANG_VX", "TORSO_ANG_VY", "TORSO_ANG_VZ",
"HIP1_ANG_V", "ANKLE1_ANG_V", "HIP2_ANG_V", "ANKLE2_ANG_V",
"HIP3_ANG_V", "ANKLE3_ANG_V", "HIP4_ANG_V", "ANKLE5_ANG_V",]


sns.set_style("dark")

parser = argparse.ArgumentParser()
parser.add_argument("--direction", type=int, default=0)
args = parser.parse_args()

direction = args.direction

fname = f'new_ppo_ant_direction{direction}'
np_file = os.path.join("data", f"{fname}.npy")
data = np.load(np_file)
limit = 300
fixed_labels = labels[-data.shape[0]:]
plt.figure(figsize=(20, 20))
for i in range(data.shape[0]):
    plt.subplot(6, 6, i + 1)
    vec = data[i, :limit]
    # vec = lowpass_filter_data(vec, 20, 1)
    # vec = smooth_exponential(vec, 0.2)
    plt.plot(vec, color='b')
    # if i < 15:
    #         grad = np.gradient(vec)
    #         plt.plot(grad, color='r')
    plt.title(fixed_labels[i])
    plt.xlim(0, limit)
    plt.xlabel("Steps")
plt.suptitle(f"Direction {direction}")
plt.tight_layout()
vis_file = os.path.join("visualizations", f"{fname}.jpg")
print(f"Saving image to {vis_file}")
plt.savefig(vis_file, dpi=120, bbox_inches="tight")
plt.close()