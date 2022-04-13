import os
import drloco
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from gym.envs.mujoco import AntEnv
from drloco.mujoco.ant import NewAntEnv
from drloco.common.utils import set_seed
from stable_baselines3.common.vec_env import VecCheckNan, VecNormalize
import torch


env_model="Ant-v2"
algo = "PPO"
logdir="logs"
seed=0

set_seed(seed, torch.cuda.is_available())
torch.autograd.set_detect_anomaly(True)

use_make=True

if not use_make:
    extra_args="import" + str(np.random.randint(5))
else:
    extra_args="gym_make_final_ant"
    extra_args="gym_make_orig"

print("Algorithm: ", algo)
tag_name = os.path.join(f"{env_model}", f"{algo}_{extra_args}")
print("Run Name: ", tag_name)

log_dir = os.path.join(logdir, tag_name, f"seed{str(seed)}")
model_dir = os.path.join(log_dir, "models")
tbdir = os.path.join(log_dir, "tb_logs")
mon_dir = os.path.join(log_dir, "gym")
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(mon_dir, exist_ok=True)
checkpoint_callback = CheckpointCallback(100000, model_dir, tag_name, 2)

callback = CallbackList([checkpoint_callback])
# Parallel environments

if not use_make:
    make_env = lambda: NewAntEnv
    env = make_vec_env(make_env, n_envs=1)
    # import pdb; pdb.set_trace()
else:
    # env = make_vec_env("NewAnt-v2", n_envs=1, monitor_dir=mon_dir)
    env = make_vec_env("Ant-v2", n_envs=1, monitor_dir=mon_dir)

training_timesteps = int(2e6)
# if os.path.exists(os.path.join(log_dir, "vec_normalize.pkl")):
#     print("Found VecNormalize Stats. Using stats")
#     env = VecNormalize.load(os.path.join(log_dir, "vec_normalize.pkl"), env)
# else:
#     print("No previous stats found. Using new VecNormalize instance.")
#     env = VecNormalize(env)


env = VecNormalize(env)
env = VecCheckNan(env, raise_exception=True)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tbdir, seed=seed)

model.learn(total_timesteps=training_timesteps)
# model.save("ppo_ant")

# del model # remove to demonstrate saving and loading

# model = PPO.load("ppo_cartpole")

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()