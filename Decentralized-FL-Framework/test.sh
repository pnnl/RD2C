#!/usr/bin/env bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling

#SBATCH -A born669
#SBATCH -t 45
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -J fed_test
#SBATCH -o test.out
#SBATCH -e error.log
#SBATCH --gres=gpu:3
#SBATCH --ntasks=10
#SBATCH --mem 64gb         # memory required by job; if unit is not specified MB will be assumed
#SBATCH --nice=0
#SBATCH --mail-type=END   # Valid type values are NONE, BEGIN, END, FAIL, REQUEUE

module load openmpi
module load cuda/11.1.1

mpirun -np 8 python main.py