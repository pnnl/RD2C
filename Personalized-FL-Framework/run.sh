#!/usr/bin/env bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling

#SBATCH -A rd2c
#SBATCH -p dl
#SBATCH -t 2
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:1
#SBATCH -J fed_test
#SBATCH -o test.out
#SBATCH -e error.log

module load python/3.7.0

srun python test.py