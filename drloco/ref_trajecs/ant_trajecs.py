import numpy as np
import scipy.io as spio
from drloco.common.utils import get_project_path
from drloco.ref_trajecs.base_ref_trajecs import BaseReferenceTrajectories

# create index constants for easier access to specific reference trajectory parts
TORSO_Z = 0
TORSO_ANGX, TORSO_ANGY, TORSO_ANGZ, TORSO_ANGW = 1, 2, 3, 4
HIP1_ANG, ANKLE1_ANG, HIP2_ANG, ANKLE2_ANG = 5, 6, 7, 8
HIP3_ANG, ANKLE3_ANG, HIP4_ANG, ANKLE5_ANG = 9, 10, 11, 12
TORSO_VX, TORSO_VY, TORSO_VZ = 13, 14, 15
TORSO_ANG_VX, TORSO_ANG_VY, TORSO_ANG_VZ = 16, 17, 18
HIP1_ANG_V, ANKLE1_ANG_V, HIP2_ANG_V, ANKLE2_ANG_V = 19, 20, 21, 22
HIP3_ANG_V, ANKLE3_ANG_V, HIP4_ANG_V, ANKLE5_ANG_V = 23, 24, 25, 26


class AntExpertTrajectories(BaseReferenceTrajectories):
    def __init__(self, qpos_indices, qvel_indices, adaptations, direction):
        # the mocaps were sampled with 500Hz
        sampling_frequency = 500 #TODO: not sure!
        # for control frequency, use the one specified in the config file
        from drloco.config.config import CTRL_FREQ
        control_frequency = CTRL_FREQ
        self.direction = direction
        # initialize the base class
        super(AntExpertTrajectories, self).__init__(sampling_frequency,
                                                          control_frequency,
                                                          qpos_indices, qvel_indices,
                                                          adaptations=adaptations)

    def _load_ref_trajecs(self):
        dir_path = get_project_path()
        file_path = 'data/new_ppo_ant_direction{}.npy'.format(self.direction)
        data = np.load(dir_path + file_path)
        return data, data

    def _get_COM_Z_pos_index(self):
        return TORSO_Z

    def is_step_left(self):
        return False

    def get_random_init_state(self):
        '''
        Random State Initialization (cf. DeepMimic Paper).
        :returns qpos and qvel of a random position on the reference trajectories
        '''
        self._pos = np.random.randint(0, self._trajec_len)
        qpos = self.get_qpos()
        qvel = self.get_qvel()
        if qpos.shape[0] != 15:
            n = qpos.shape[0]
            xy = np.random.uniform(size=15 - n, low=-0.1, high=0.1)
            qpos = np.concatenate((xy, qpos), 0)
        return qpos, qvel

    def get_desired_walking_velocity_vector(self, do_eval, debug=False):
        if False and do_eval:
            # during evaluation, let the agent walk just straight.
            # This way, we can retain our current evaluation metrics.
            return [1.2, 0.0]

        # get the average velocities in x and z directions
        # average over n seconds
        n_seconds = 0.5
        n_timesteps = int(n_seconds * self._sample_freq)
        # consider the reference trajectory has a maximum length
        end_pos = min(self._pos + n_timesteps, self._trajec_len-1)
        qvels_x = self._qvel_full[TORSO_ANG_VX, self._pos: end_pos]
        # NOTE: y direction in the simulation corresponds to z direction in the mocaps
        qvels_y = self._qvel_full[TORSO_ANG_VY, self._pos: end_pos]
        # get the mean velocities
        mean_x_vel = np.mean(qvels_x)
        mean_y_vel = np.mean(qvels_y)
        if debug:
            try:
                self.des_vels_x += [mean_x_vel]
                self.des_vels_y += [mean_y_vel]
                # calculate the walker position by integrating the velocity vector
                self.xpos += mean_x_vel * 1/self._control_freq
                self.ypos += mean_y_vel * 1/self._control_freq
                self.xposs += [self.xpos]
                self.yposs += [self.ypos]
            except:
                self.des_vels_x = [mean_x_vel]
                self.des_vels_y = [mean_y_vel]
                self.xpos, self.ypos = 0, 0
                self.xposs = [self.xpos]
                self.yposs = [self.ypos]

            if len(self.des_vels_x) > 1000:
                from matplotlib import pyplot as plt
                fig, subs = plt.subplots(1,4)
                subs[0].plot(self.des_vels_x)
                subs[1].plot(self.des_vels_y)
                subs[2].plot(self.des_vels_x, self.des_vels_y)
                subs[3].plot(self.xposs, self.yposs)
                for i in range(3):
                    subs[i].set_title('Desired Velocity\nin {} direction'.format(['X', 'Y', 'X and Y'][i]))
                subs[3].set_title('Walkers position\n(from mean des_vel)')
                plt.show()
                exit(33)
        return [mean_x_vel, mean_y_vel]
