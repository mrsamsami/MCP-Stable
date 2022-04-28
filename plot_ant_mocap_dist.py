import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

labels = [
    "TORSO_X", "TORSO_Y", "TORSO_Z",
    "TORSO_ANGX", "TORSO_ANGY", "TORSO_ANGZ", "TORSO_ANGW",
    "HIP1_ANG", "ANKLE1_ANG", "HIP2_ANG", "ANKLE2_ANG",
    "HIP3_ANG", "ANKLE3_ANG", "HIP4_ANG", "ANKLE5_ANG",
    "TORSO_VX", "TORSO_VY", "TORSO_VZ",
    "TORSO_ANG_VX", "TORSO_ANG_VY", "TORSO_ANG_VZ",
    "HIP1_ANG_V", "ANKLE1_ANG_V", "HIP2_ANG_V", "ANKLE2_ANG_V",
    "HIP3_ANG_V", "ANKLE3_ANG_V", "HIP4_ANG_V", "ANKLE5_ANG_V",
]

sns.set_style("dark")

parser = argparse.ArgumentParser()
parser.add_argument("--direction", type=int, default=0)
args = parser.parse_args()

direction = args.direction

fname = f'new_ppo_ant_direction{direction}'
np_file = os.path.join("data", f"{fname}.npy")
data = np.load(np_file)
limit = 300
fixed_labels = labels[-data.shape[0]:]
plt.figure(figsize=(20, 20))
for i in range(data.shape[0]):
    plt.subplot(6, 6, i + 1)
    vec = data[i, :limit]
    plt.plot(vec, color='b')
    plt.title(fixed_labels[i])
    plt.xlim(0, limit)
    plt.xlabel("Steps")
plt.suptitle(f"Direction {direction}")
plt.tight_layout()
vis_file = os.path.join("visualizations", f"{fname}.jpg")
print(f"Saving image to {vis_file}")
plt.savefig(vis_file, dpi=120, bbox_inches="tight")
plt.close()