import numpy as np
from drloco.config import hypers as cfg
from drloco.common.utils import get_project_path, is_remote
from drloco.mujoco.mimic_env import MimicEnv
from drloco.ref_trajecs import ant_trajecs as refs

# pause sim on startup to be able to change rendering speed, camera perspective etc.
pause_mujoco_viewer_on_start = True and not is_remote()

# qpos and qvel indices for quick access to the reference trajectories
qpos_indices = [refs.TORSO_Z, refs.TORSO_ANGX, refs.TORSO_ANGY, refs.TORSO_ANGZ, refs.TORSO_ANGW,
refs.HIP1_ANG, refs.ANKLE1_ANG, refs.HIP2_ANG, refs.ANKLE2_ANG,
refs.HIP3_ANG, refs.ANKLE3_ANG, refs.HIP4_ANG, refs.ANKLE5_ANG]

qvel_indices = [refs.TORSO_VX, refs.TORSO_VY, refs.TORSO_VZ,
refs.TORSO_ANG_VX, refs.TORSO_ANG_VY, refs.TORSO_ANG_VZ,
refs.HIP1_ANG_V, refs.ANKLE1_ANG_V, refs.HIP2_ANG_V, refs.ANKLE2_ANG_V,
refs.HIP3_ANG_V, refs.ANKLE3_ANG_V, refs.HIP4_ANG_V, refs.ANKLE5_ANG_V]

ref_trajec_adapts = {}

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
        ref_t = np.random.randint(4)
        # import pdb; pdb.set_trace()
        MimicEnv.__init__(self, mujoco_xml_file, self.possible_reference_trajectories[ref_t])

    # def viewer_setup(self):
    #     self.viewer.cam.trackbodyid = 2
    #     self.viewer.cam.distance = self.model.stat.extent * 0.5
    #     self.viewer.cam.lookat[2] = 1.15
    #     self.viewer.cam.elevation = -20

    # ----------------------------
    # Methods we override:
    # ----------------------------

    def reset(self):
        ref_t = np.random.randint(4)
        self.refs = self.possible_reference_trajectories[ref_t]
        return super().reset()

    def _get_COM_indices(self):
        return [refs.TORSO_Z]

    def get_joint_indices_for_phase_estimation(self):
        # return both knee and hip joints
        return [refs.ANKLE1_ANG, refs.ANKLE2_ANG, refs.ANKLE3_ANG, refs.ANKLE5_ANG]

    # def has_ground_contact(self):
    #     has_contact = [False, False]
    #     for contact in self.data.contact[:self.data.ncon]:
    #         if contact.geom1 == 0 and contact.geom2 == 4:
    #             # right foot has ground contact
    #             has_contact[1] = True
    #         elif contact.geom1 == 0 and contact.geom2 == 7:
    #             # left foot has ground contact
    #             has_contact[0] = True
    #     return has_contact
