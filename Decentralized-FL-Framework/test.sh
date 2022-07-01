#!/usr/bin/env bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling

#SBATCH -A rd2c
#SBATCH -p dl
#SBATCH -t 40
#SBATCH -N 1
#SBATCH -n 6
#SBATCH --gres=gpu:2
#SBATCH -J fed_test

module purge
module load python/3.7.0
module load cuda/11.1
source born669/venv_fed/bin/activate
module load gcc
module load openmpi/4.1.4

mpirun -np 6 python main.py