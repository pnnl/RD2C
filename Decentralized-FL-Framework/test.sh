#!/usr/bin/env bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling

#SBATCH -A rd2c
#SBATCH -p dl
#SBATCH -t 40
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:3
#SBATCH -J fed_test

module purge
module load python/3.7.0
module load gcc
module load openmpi

mpirun -np 8 python main.py