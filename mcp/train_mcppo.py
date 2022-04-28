import copy
import os

import gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecCheckNan, VecNormalize

import env
from mcp.policies import MPPO
from mcp.utils import SaveVideoCallback

env_model = "NewRandomGoalAnt-v2"

direction = 0
env_direction = {
    0: 0,
    1: 180,
    2: 90,
    3: 270,
}

learn_log_std = False
big_model = False
assert not learn_log_std
assert not big_model

run_id = f"transfer_mcppo"
direction = direction
algo = "PPO"
logdir = "logs"
seed = 0
vec_normalise = False

num_envs = 4
training_timesteps = int(3e6)
checkpoint_freq = 200000
eval_freq = 50000
video_freq = 100000

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

env_kwargs = {
    "direction": env_direction[direction],
    "direction_range": [270, 360],
}

assert "GoalAnt-v2" in env_model
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

custom_objects = {"lr_schedule": lambda x: 0.003, "clip_range": lambda x: 0.02}

models = [
    PPO.load(
        f"logs/NewAnt-v2/PPO_vec_norm_False_direction_{dir}_5M/seed0/final.zip",
        custom_objects=custom_objects,
    )
    for dir in range(4)
]
models = [mod.policy for mod in models]

policy_kwargs = {
    "state_dim": env.observation_space.shape[0] - 2,
    "goal_dim": 2,
    "models": copy.deepcopy(models),
}
mppo_model = PPO(
    MPPO, env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log=tbdir, seed=seed
)

mppo_model.learn(total_timesteps=training_timesteps, callback=callback)

checkpoint_dir = os.path.join(log_dir, "final")
mppo_model.save(checkpoint_dir)
