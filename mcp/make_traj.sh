#!/bin/bash

source ~/ENV/bin/activate

run_ids="PPO_mcp_full_numprim_8_use_mcp_hyper_False"
labels="PPO_mcp_full_numprim_8_use_mcp_hyper_False MCP"

dirs_vector="135"
python make_traj_plot.py --run_ids ${run_ids} --labels ${labels} --dirs_vector ${dirs_vector} --title Pretraining

# dirs_vector="90"
# python make_traj_plot.py --run_ids ${run_ids} --labels ${labels} --dirs_vector ${dirs_vector}

# dirs_vector="135"
# python make_traj_plot.py --run_ids ${run_ids} --labels ${labels} --dirs_vector ${dirs_vector}

# dirs_vector="180"
# python make_traj_plot.py --run_ids ${run_ids} --labels ${labels} --dirs_vector ${dirs_vector}

# dirs_vector="225"
# python make_traj_plot.py --run_ids ${run_ids} --labels ${labels} --dirs_vector ${dirs_vector}

run_ids="PPO_scratch PPO_transfer_mcp_full_numprim_8_use_mcp_hyper_False PPO_transfer_mcppo"
labels="PPO_scratch Scratch PPO_transfer_mcp_full_numprim_8_use_mcp_hyper_False MCP(p=8) PPO_transfer_mcppo MCPPO"

dirs_vector="315"
python make_traj_plot.py --run_ids ${run_ids} --labels ${labels} --dirs_vector ${dirs_vector} --title Transfer


# "PPO_mcp_full_numprim_8_use_mcp_hyper_False PPO_scratch PPO_transfer_mcp_full_numprim_8_use_mcp_hyper_False PPO_transfer_mcppo PPO_mcppo_sb3log_std_False_8directions_6M PPO_test_mcp_naive_np_8_sb3log_std_False_unit_test"