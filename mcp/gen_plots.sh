#!/bin/bash

source ~/ENV/bin/activate

dirs_vector="135"
python gen_trajs.py --run_id PPO_mcp_full_numprim_8_use_mcp_hyper_False --dirs_vector ${dirs_vector}

dirs_vector="315"
python gen_trajs.py --run_id PPO_scratch --dirs_vector ${dirs_vector}
python gen_trajs.py --run_id PPO_transfer_mcppo --dirs_vector ${dirs_vector}
python gen_trajs.py --run_id PPO_transfer_mcp_full_numprim_8_use_mcp_hyper_False --dirs_vector ${dirs_vector}
