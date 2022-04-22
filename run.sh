#!/bin/bash

source ~/ENV/bin/activate

direction=${SLURM_ARRAY_TASK_ID}
# direction=0
vec_normalise=False
ID="vec_norm_${vec_normalise}_direction_${direction}_5M"
python drloco/train_ant.py --direction ${direction} --id ${ID} --vec_normalise ${vec_normalise}
# python drloco/test_ant.py

# sbatch --time=1:40:00 --mem=48G --account=rrg-ebrahimi --cpus-per-task=8 --array=0-3 --job-name=ant_direction --out=out/%x_%A_%a.out run.sh