#!/bin/bash

source ~/ENV/bin/activate

run_ids="PPO_mcppo_sb3log_std_False_8directions_6M PPO_test_mcp_naive_np_8_sb3log_std_False_unit_test"
labels="PPO_mcppo_sb3log_std_False_8directions_6M MCPPO PPO_test_mcp_naive_np_8_sb3log_std_False_unit_test MCP(p=8)"

dirs_vectors="0 45 90 135 180 225 270 315"
for dirs_vector in $dirs_vectors
do
    python make_traj_plot.py --run_ids ${run_ids} --labels ${labels} --dirs_vector ${dirs_vector}
done

# dirs_vector="45"
# python make_traj_plot.py --run_ids ${run_ids} --labels ${labels} --dirs_vector ${dirs_vector}

# dirs_vector="90"
# python make_traj_plot.py --run_ids ${run_ids} --labels ${labels} --dirs_vector ${dirs_vector}

# dirs_vector="135"
# python make_traj_plot.py --run_ids ${run_ids} --labels ${labels} --dirs_vector ${dirs_vector}

# dirs_vector="180"
# python make_traj_plot.py --run_ids ${run_ids} --labels ${labels} --dirs_vector ${dirs_vector}

# dirs_vector="225"
# python make_traj_plot.py --run_ids ${run_ids} --labels ${labels} --dirs_vector ${dirs_vector}

# dirs_vector="315"
# python make_traj_plot.py --run_ids ${run_ids} --labels ${labels} --dirs_vector ${dirs_vector}
