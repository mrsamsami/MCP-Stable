import numpy as np
from drloco.config import hypers as cfg
from drloco.common.utils import get_project_path, is_remote
from drloco.mujoco.mimic_env import MimicEnv
from drloco.ref_trajecs import ant_trajecs as refs

# pause sim on startup to be able to change rendering speed, camera perspective etc.
pause_mujoco_viewer_on_start = True and not is_remote()


qpos_indices = range(15)
qvel_indices = range(15, 29)

ref_trajec_adapts = {}
randomise_traj = False
class MimicAntEnv(MimicEnv):
    '''
    The 2D Mujoco Walker from OpenAI Gym extended to match
    the 3D bipedal walker model from Guoping Zhao.
    '''

    def __init__(self):
        # init reference trajectories
        # by specifying the indices in the mocap data to use for qpos and qvel
        self.possible_reference_trajectories = [refs.AntExpertTrajectories(qpos_indices, qvel_indices, {}, direction=i) for i in range(4)]
        mujoco_xml_file = 'ant.xml'
        # init the mimic environment
        if randomise_traj:
            ref_t = np.random.randint(4)
        else:
            ref_t = 0
        MimicEnv.__init__(self, mujoco_xml_file, self.possible_reference_trajectories[ref_t])

    def reset(self):
        if randomise_traj:
            ref_t = np.random.randint(4)
        else:
            ref_t = 0
        self.refs = self.possible_reference_trajectories[ref_t]
        return super().reset()

    def _get_COM_indices(self):
        return [0, 1, 2]

    def get_joint_indices_for_phase_estimation(self):
        # return both knee and hip joints
        return [8, 10, 12, 14], [7, 9, 11, 13]

