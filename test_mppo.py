import sys
import os
sys.path.append(os.getcwd())

import numpy as np
from tqdm import tqdm
import env
import gym
import sys
from drloco.common import utils
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--direction", type=int, default=0)
args = parser.parse_args()

direction = args.direction
run_id = "PPO_mcppo_8directions"
run_id = "PPO_mcppo_logstd_8directions_6M"
save_gif = True
eval_env = make_vec_env(env.GoalAnt, n_envs=1, env_kwargs={'direction': direction})
model_path = f'logs/NewRandomGoalAnt-v2/{run_id}/seed0/final.zip'
model = PPO.load(model_path)
save_path = "mcppo"
os.makedirs(save_path, exist_ok=True)

obs = eval_env.reset()
img = eval_env.render("rgb_array")
imgs = [img]
done = False
tot_r = 0.0
weights=[]
print(f"Begin Evaluation")
ep_max_len = 1000
pbar = tqdm(total=ep_max_len)
i = 0
dirs = np.array([0, 45, 90, 135, 180, 225, 270, 315])
dirs = np.array([0, 90, 180, 270])
dirs = dirs[..., None]
dirs = np.repeat(dirs, ep_max_len/dirs.shape[0], 1)
dirs = dirs.flatten()
# dirs = np.concatenate((np.linspace(0, 180, ep_max_len // 2), np.linspace(180, 0, ep_max_len // 2)))
# dirs = np.round(dirs, 1)
while not done:
    eval_env.envs[0].set_direction(dirs[i])
    action, _ = model.predict(obs, deterministic=True)
    weight = model.policy.predict_weights(obs)
    weights.append(weight)
    obs, reward, done, info = eval_env.step(action)
    img = eval_env.render("rgb_array")
    imgs.append(img)
    tot_r += reward
    pbar.update(1)
    i+= 1
    if i > ep_max_len - 1:
        done = True
pbar.close()
print(f"Evaluation Reward: {tot_r}")
weights = np.array(weights).squeeze(1)
fname=os.path.join(save_path, f"{run_id}_weights.npy")
np.save(fname, weights)
ep_len = weights.shape[0]
print(f"Ep Len: {ep_len}")
for i in range(weights.shape[1]):
    plt.plot(weights[:, i], label=f"Model {i}")
plt.xlim(0, ep_len)
plt.ylim(0, 1)
plt.title(f"Weights assigned to PPO primitives {eval_env.envs[0].direction}")
plt.tight_layout()
plt.legend()
fname=os.path.join(save_path, f"{run_id}_weights.jpg")
plt.savefig(fname, bbox_inches="tight", dpi=120)
plt.close()
imgs = np.array(imgs)
fname=os.path.join(save_path, f"{run_id}_eval_video.gif")
fps = 30 if ep_len < 200 else 60
utils.write_gif_to_disk(imgs, fname, fps)
