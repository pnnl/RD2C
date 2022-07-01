#!/usr/bin/env bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling

#SBATCH -A rd2c
#SBATCH -p gpu
#SBATCH -t 2
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
# #SBATCH --gres=gpu:0
#SBATCH -J fed_test
#SBATCH -o test.out
#SBATCH -e error.log

module load python/3.7.2

srun python test.py