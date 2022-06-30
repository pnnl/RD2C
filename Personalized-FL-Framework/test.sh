#!/usr/bin/env bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling

#SBATCH -A r2dc
#SBATCH -t 2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J fed_test
#SBATCH -o test.out
#SBATCH -e error.log
#SBATCH --gres=gpu:1

# module load openmpi
# module load cuda/11.1.1

python test.py