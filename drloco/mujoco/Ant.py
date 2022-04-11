import numpy as np
from gym.envs.mujoco import AntEnv


class AntEnvV2(AntEnv):
    def __init__(self, direction: int = 0):
        super.__init__()
        self.direction = direction

    def step(self, a):
        if self.direction == 0:
            xposbefore = self.get_body_com("torso")[0]
            self.do_simulation(a, self.frame_skip)
            xposafter = self.get_body_com("torso")[0]

        elif self.direction == 1:
            xposafter = self.get_body_com("torso")[0]
            self.do_simulation(a, self.frame_skip)
            xposbefore = self.get_body_com("torso")[0]

        elif self.direction == 2:
            xposbefore = self.get_body_com("torso")[1]
            self.do_simulation(a, self.frame_skip)
            xposafter = self.get_body_com("torso")[1]

        elif self.direction == 3:
            xposafter = self.get_body_com("torso")[1]
            self.do_simulation(a, self.frame_skip)
            xposbefore = self.get_body_com("torso")[1]

        forward_reward = (xposafter - xposbefore) / self.dt
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