import sys
import os
sys.path.append(os.getcwd())

import numpy as np

import sys

import argparse
import matplotlib.pyplot as plt

palette= (
        "#0022ff",
        "#33aa00",
        "#ff0011",
        "#ddaa00",
        "#cc44dd",
        "#0088aa",
        "#001177",
        "#117700",
        "#990022",
        "#885500",
        "#553366",
        "#006666",
    )

parser = argparse.ArgumentParser()
parser.add_argument("--run_ids", nargs="+", default=["PPO_mcppo_5directions"])
parser.add_argument("--traj_dir", type=str, default="trajs")
parser.add_argument("--dirs_vector", type=str, default="0")
parser.add_argument("--labels", nargs="+", default=None)
parser.add_argument("--title", type=str, default=None)
args = parser.parse_args()

if args.labels:
    assert len(args.labels) % 2 == 0
    args.labels = {k: v for k, v in zip(args.labels[:-1], args.labels[1:])}

logdir = args.traj_dir
save_dir = "plots"
name = "traj"
save_path = os.path.join(save_dir, name)
os.makedirs(save_path, exist_ok=True)

ep_max_len = 500

# ! Cardinal Direction Heading
dirs_vector = args.dirs_vector
dirs_vector = [int(d) for d in dirs_vector.split(",")]
dirs_vector = np.array(dirs_vector)
dirs = dirs_vector[..., None]
dirs = np.repeat(dirs, ep_max_len/dirs.shape[0], 1)
dirs = dirs.flatten()

max_dist = 50
dir_rad = np.round(dirs / 180 * np.pi, 4)
dir_vector = (np.cos(dir_rad), np.sin(dir_rad))
dir_vector = np.round(dir_vector, 2)
vec = np.linspace(0, max_dist * 2, ep_max_len)
vec = np.stack([vec, vec])
dir_vector = vec * dir_vector

traj_name = "_".join([str(dd) for dd in dirs_vector])

run_ids = args.run_ids
paths = [os.path.join(logdir, r) for r in run_ids]

colors = {}
for i, run_id in enumerate(run_ids):
    colors[run_id] = palette[i % len(palette)]

for path, run_id in zip(paths, run_ids):
    fname = os.path.join(path, f"{traj_name}_run_data.npz")
    if not os.path.exists(fname):
        continue
    data = np.load(fname)
    data = np.stack([v for v in data.values()])
    n_runs = data.shape[0]

    if args.labels and run_id in args.labels:
        agent = args.labels[run_id]
    else:
        agent = run_id.replace("_", " ").title()

    for i in range(n_runs):
        run_data = np.array(data[i]).T
        line = plt.plot(run_data[0], run_data[1], color=colors[run_id], alpha=0.5)
        if i == n_runs - 1:
            line[0].set_label(f'{agent}')

plt.plot(dir_vector[0], dir_vector[1], linestyle="dashed", color="black", alpha=0.7, label="Target")
plt.xlim(-max_dist, max_dist)
plt.ylim(-max_dist, max_dist)
plt.xlabel("x (meters)")
plt.xlabel("y (meters)")

if args.title is None:
    title = f"{traj_name} Trajectory"
else:
    title = args.title

plt.title(f"{title}")
plt.tight_layout()
plt.legend()

fname=os.path.join(save_path, f"{traj_name}_traj.jpg")
plt.savefig(fname, bbox_inches="tight", dpi=200)
plt.close()
print(f"Plot saved to {fname}")


