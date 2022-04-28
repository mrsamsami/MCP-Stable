import argparse
import os

import gym
import numpy as np
import torch
from drloco.common.utils import set_seed
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecCheckNan

import env
from utils import SaveVideoCallback

env_model = "NewAnt-v2"

parser = argparse.ArgumentParser()
parser.add_argument("--direction", type=int, default=3)
parser.add_argument("--id", type=str, default="baseline")
parser.add_argument("--algo", type=str, default="PPO")
parser.add_argument("--logdir", type=str, default="logs")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--vec_normalise", type=str, default="False")
args = parser.parse_args()


env_model = "NewRandomGoalAnt-v2"

direction = 0
env_direction = {
    0: 0,
    1: 180,
    2: 90,
    3: 270,
}

run_id = args.id
direction = args.direction
algo = args.algo
logdir = args.logdir
seed = args.seed
vec_normalise = args.vec_normalise == "True"
num_envs = 4
training_timesteps = int(4.5e6)
checkpoint_freq = 500000
eval_freq = 50000
video_freq = 100000

set_seed(seed, torch.cuda.is_available())
torch.autograd.set_detect_anomaly(True)

print("Algorithm: ", algo)
tag_name = os.path.join(f"{env_model}", f"{algo}_{run_id}")
print("Run Name: ", tag_name)

log_dir = os.path.join(logdir, tag_name, f"seed{str(seed)}")
model_dir = os.path.join(log_dir, "models")
tbdir = os.path.join(log_dir, "tb_logs")
mon_dir = os.path.join(log_dir, "gym")

os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(mon_dir, exist_ok=True)

env_kwargs = {"direction": env_direction[direction], "direction_range": [270, 360]}

env = make_vec_env(
    env_model, n_envs=num_envs, monitor_dir=mon_dir, env_kwargs=env_kwargs, seed=seed
)

assert not vec_normalise
print("Not using VecNormalize")

env = VecCheckNan(env, raise_exception=True)

checkpoint_callback = CheckpointCallback(
    int(checkpoint_freq // num_envs), model_dir, tag_name, 2
)
eval_env = make_vec_env(env_model, n_envs=1, monitor_dir=mon_dir, env_kwargs=env_kwargs)

save_video_callback = SaveVideoCallback(
    eval_env,
    int(eval_freq // num_envs),
    int(video_freq // num_envs),
    vec_normalise,
    log_dir,
    2,
)
callback = CallbackList([checkpoint_callback, save_video_callback])

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tbdir, seed=seed)

model.learn(total_timesteps=training_timesteps, callback=callback)

checkpoint_dir = os.path.join(log_dir, "final")
model.save(checkpoint_dir)
