#!/usr/bin/env bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling

#SBATCH -A rd2c
#SBATCH -p dl
#SBATCH -t 40
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH -J fed_test
#SBATCH -o results.out
#SBATCH -e error.log

module load cuda
module load python/3.7.0

srun python test.py