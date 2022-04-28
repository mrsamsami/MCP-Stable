import gym
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.envs.registration import register


class DirAntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, direction=0):
        """
        direction: angle in degrees, between 0 and 360 used to specify the desired heading of the agent. Measured anti-clockwise
        """
        self.set_direction(direction)
        mujoco_env.MujocoEnv.__init__(self, "ant.xml", 5)
        utils.EzPickle.__init__(self)

    def set_direction(self, direction):
        assert 0 <= direction <= 360
        self.direction = direction
        direction = self.direction / 180 * np.pi
        self.desired_heading = np.round((np.cos(direction), np.sin(direction)), 3)

    def add_goal(self, obs):
        obs = np.concatenate((obs, self.desired_heading), 0)
        return obs

    def step(self, a):
        xy_position_before = self.get_body_com("torso")[:2].copy()
        self.do_simulation(a, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt

        forward_reward = np.dot(xy_velocity, self.desired_heading)

        ctrl_cost = 0.5 * np.square(a).sum()
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return (
            ob,
            reward,
            done,
            dict(
                reward_forward=forward_reward,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
            ),
        )

    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat,
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            ]
        )

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5


class GoalAnt(DirAntEnv):
    def __init__(self, direction=0):
        super(GoalAnt, self).__init__(direction)
        # NOTE TO SELF: MujocoEnv calls env.step and uses the returned obs to set the observation_space, hence there is no need to manually change the observation space here.

    def step(self, a):
        obs, reward, done, info = super().step(a)
        return self.add_goal(obs), reward, done, info

    def reset(self):
        obs = super().reset()
        return self.add_goal(obs)


class RandomGoalAnt(DirAntEnv):
    def __init__(self, direction=0, direction_range=None, direction_list=[0, 90, 180, 270]):
        self.direction_range = direction_range
        self.direction_list = direction_list
        super(RandomGoalAnt, self).__init__(direction)
        # NOTE TO SELF: MujocoEnv calls env.step and uses the returned obs to set the observation_space, hence there is no need to manually change the observation space here.

    def step(self, a):
        obs, reward, done, info = super().step(a)
        return self.add_goal(obs), reward, done, info

    def reset(self, direction=None):
        if direction is None:
            if self.direction_range is not None:
                direction = np.random.randint(
                    self.direction_range[0], self.direction_range[1]
                )
            elif self.direction_list is not None:
                direction = np.random.choice(self.direction_list)
        self.set_direction(direction)
        obs = super().reset()
        return self.add_goal(obs)


register(
    id="NewAnt-v2",
    entry_point=DirAntEnv,
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id="NewGoalAnt-v2",
    entry_point=GoalAnt,
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id="NewRandomGoalAnt-v2",
    entry_point=RandomGoalAnt,
    max_episode_steps=1000,
    reward_threshold=6000.0,
)


if __name__ == "__main__":
    import gym

    env = gym.make("NewAnt-v2")
    print(env.action_space.shape, env.observation_space.shape)
    print(env.reset().shape, env.step(env.action_space.sample())[0].shape)

    env = gym.make("NewGoalAnt-v2")
    print(env.action_space.shape, env.observation_space.shape)
    print(env.reset().shape, env.step(env.action_space.sample())[0].shape)

    env = gym.make("NewRandomGoalAnt-v2")
    print(env.action_space.shape, env.observation_space.shape)
    print(env.reset().shape, env.step(env.action_space.sample())[0].shape)
