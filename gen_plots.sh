#!/bin/bash

source ~/ENV/bin/activate

dirs_vectors="0"
# run_ids="PPO_mcppo_5directions PPO_mcppo_8directions PPO_mcp_naive_sb3log_std_False_unit_test PPO_mcp_naive_sb3log_std_True_unit_test PPO_mcppo_sb3log_std_False_8directions_6M PPO_mcppo_sb3log_std_True_8directions_6M PPO_test_mcp_naive_np_8_sb3log_std_False_unit_test PPO_test_mcp_naive_np_8_sb3log_std_True_unit_test"
run_ids="PPO_mcppo_sb3log_std_False_8directions_6M PPO_test_mcp_naive_np_8_sb3log_std_False_unit_test"
for dirs_vector in $dirs_vectors
    do
    for run_id in $run_ids
    do
        python test_mppo.py --run_id ${run_id} --dirs_vector ${dirs_vector}
    done
done
# dirs_vector="45"

# python test_mppo.py --run_id PPO_mcppo_5directions --dirs_vector ${dirs_vector}
# python test_mppo.py --run_id PPO_mcppo_8directions --dirs_vector ${dirs_vector}

# python test_mppo.py --run_id PPO_mcp_naive_sb3log_std_False_unit_test --dirs_vector ${dirs_vector}
# python test_mppo.py --run_id PPO_mcp_naive_sb3log_std_True_unit_test --dirs_vector ${dirs_vector}

# python test_mppo.py --run_id PPO_mcppo_sb3log_std_False_8directions_6M --dirs_vector ${dirs_vector}
# python test_mppo.py --run_id PPO_mcppo_sb3log_std_True_8directions_6M --dirs_vector ${dirs_vector}

# python test_mppo.py --run_id PPO_test_mcp_naive_np_8_sb3log_std_False_unit_test --dirs_vector ${dirs_vector}
# python test_mppo.py --run_id PPO_test_mcp_naive_np_8_sb3log_std_True_unit_test --dirs_vector ${dirs_vector}

# dirs_vector="225"
# python test_mppo.py --run_id PPO_mcppo_5directions --dirs_vector ${dirs_vector}
# python test_mppo.py --run_id PPO_mcppo_8directions --dirs_vector ${dirs_vector}

# python test_mppo.py --run_id PPO_mcp_naive_sb3log_std_False_unit_test --dirs_vector ${dirs_vector}
# python test_mppo.py --run_id PPO_mcp_naive_sb3log_std_True_unit_test --dirs_vector ${dirs_vector}

# python test_mppo.py --run_id PPO_mcppo_sb3log_std_False_8directions_6M --dirs_vector ${dirs_vector}
# python test_mppo.py --run_id PPO_mcppo_sb3log_std_True_8directions_6M --dirs_vector ${dirs_vector}

# python test_mppo.py --run_id PPO_test_mcp_naive_np_8_sb3log_std_False_unit_test --dirs_vector ${dirs_vector}
# python test_mppo.py --run_id PPO_test_mcp_naive_np_8_sb3log_std_True_unit_test --dirs_vector ${dirs_vector}

# dirs_vector="315"
# python test_mppo.py --run_id PPO_mcppo_5directions --dirs_vector ${dirs_vector}
# python test_mppo.py --run_id PPO_mcppo_8directions --dirs_vector ${dirs_vector}

# python test_mppo.py --run_id PPO_mcp_naive_sb3log_std_False_unit_test --dirs_vector ${dirs_vector}
# python test_mppo.py --run_id PPO_mcp_naive_sb3log_std_True_unit_test --dirs_vector ${dirs_vector}

# python test_mppo.py --run_id PPO_mcppo_sb3log_std_False_8directions_6M --dirs_vector ${dirs_vector}
# python test_mppo.py --run_id PPO_mcppo_sb3log_std_True_8directions_6M --dirs_vector ${dirs_vector}

# python test_mppo.py --run_id PPO_test_mcp_naive_np_8_sb3log_std_False_unit_test --dirs_vector ${dirs_vector}
# python test_mppo.py --run_id PPO_test_mcp_naive_np_8_sb3log_std_True_unit_test --dirs_vector ${dirs_vector}