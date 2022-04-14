import logging
from gym.envs.registration import register

register(
    id='NewAnt-v2',
    entry_point='drloco.mujoco.ant:DirAntEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)