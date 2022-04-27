import os
import numpy as np
from tqdm import tqdm
import drloco
import gym
import sys
from drloco.common import utils
from drloco.common.utils import save_as_gif
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--direction", type=int, default=0)
args = parser.parse_args()

direction = args.direction

save_gif = True
env = make_vec_env('NewAnt-v2', n_envs=1, env_kwargs={'direction': direction})
model_path = f'logs/NewAnt-v2/PPO_vec_norm_False_direction_{direction}_5M/seed0/final.zip'
model = PPO.load(model_path)

labels = [
    "TORSO_X", "TORSO_Y", "TORSO_Z",
    "TORSO_ANGX", "TORSO_ANGY", "TORSO_ANGZ", "TORSO_ANGW",
"HIP1_ANG", "ANKLE1_ANG", "HIP2_ANG", "ANKLE2_ANG",
"HIP3_ANG", "ANKLE3_ANG", "HIP4_ANG", "ANKLE5_ANG",
"TORSO_VX", "TORSO_VY", "TORSO_VZ",
"TORSO_ANG_VX", "TORSO_ANG_VY", "TORSO_ANG_VZ",
"HIP1_ANG_V", "ANKLE1_ANG_V", "HIP2_ANG_V", "ANKLE2_ANG_V",
"HIP3_ANG_V", "ANKLE3_ANG_V", "HIP4_ANG_V", "ANKLE5_ANG_V",]

done=False
obs = env.reset()
img = env.render(mode="rgb_array")
frames = [img]
data = []
rew = []
pbar = tqdm(total=1000)
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    rew.append(rewards)
    if save_gif:
        img = env.render(mode="rgb_array")
        frames.append(img)
    state_vector = np.concatenate(
        [
            env.envs[0].sim.data.qpos.flat,
            env.envs[0].sim.data.qvel.flat,
        ]
    )
    data.append(state_vector)
    done = dones[0]
    pbar.update(1)
pbar.close()
env.close()

rew = np.array(rew)
print(rew.sum())
data = np.array(data).T
plt.figure(figsize=(20, 20))
for i in range(data.shape[0]):
    plt.subplot(6, 6, i + 1)
    plt.plot(data[i, :], color='b')
    if i < 15:
            grad = np.gradient(data[i, :])
            plt.plot(grad, color='r')
    plt.title(labels[i])
    plt.xlim(0, 300)
plt.savefig("temp.jpg", dpi=120)
plt.close()
assert data.shape[0] == 29
fname = f'new_ppo_ant_direction{direction}'
np_file = os.path.join("data", fname)
np.save(np_file, data)
if save_gif:
    video_file = os.path.join("animations", f"{fname}.gif")
    utils.write_gif_to_disk(frames, video_file, 60)