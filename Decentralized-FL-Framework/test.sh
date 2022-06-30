#!/usr/bin/env bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling

#SBATCH -A rd2c
#SBATCH -t 20
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -J fed_test
#SBATCH -o test.out
#SBATCH -e error.log
#SBATCH --gres=gpu:3

module load openmpi
module load cuda/11.1.1

mpirun -np 8 python main.py