#!/bin/bash

source ~/ENV/bin/activate

python drloco/train.py

# sbatch --time=4:40:00 --mem=48G --account=rrg-ebrahimi --cpus-per-task=8 --array=0-0 --job-name=ant_deepmimic --out=out/%x_%A_%a.out new_run.sh