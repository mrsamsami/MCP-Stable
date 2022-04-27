#!/bin/bash

source ~/ENV/bin/activate

# python drloco/train.py
python mcp_naive.py

# sbatch --time=5:59:00 --mem=48G --account=rrg-ebrahimi --cpus-per-task=8 --array=0-0 --job-name=ant_mcp_naive --out=out/%x_%A_%a.out mcp_n.sh