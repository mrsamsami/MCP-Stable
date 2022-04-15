import drloco
import gym
import sys
from drloco.common.utils import save_as_gif
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

direction = int(sys.argv[1])
save_gif = bool(sys.argv[2]) if len(sys.argv) == 3 else True
env = make_vec_env('NewAnt-v2', n_envs=1, env_kwargs={'direction': direction})
model = PPO.load("models/ppo_ant_direction{}".format(direction))
# model = PPO.load("ppo_ant1".format(direction))

obs = env.reset()
frames = []

for t in range(1000):
    if save_gif:
        frames.append(env.render(mode="rgb_array"))
    else:
        env.render()

    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)

    if dones[0]:
        break

env.close()
if save_gif:
    save_as_gif(frames, './animations/', "ppo_ant_direction{}.gif".format(direction))