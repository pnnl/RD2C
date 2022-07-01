#!/usr/bin/env bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling

#SBATCH -A rd2c
#SBATCH -p short
#SBATCH -t 2
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -J fed_test
#SBATCH -o test.out
#SBATCH -e error.log

# module load openmpi
# module load cuda/11.1.1

srun python test.py