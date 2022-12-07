#!/usr/bin/env bash
#SBATCH -A rd2c
#SBATCH -p dl
#SBATCH -t 50
#SBATCH -N 1
#SBATCH -n 10
#SBATCH --gpus-per-node=0
#SBATCH -J middle_ablation

module purge
module load python/3.7.0
module load cuda/11.1
module load gcc
module load openmpi
source /people/born669/middle/bin/activate

mpirun -n 10 python MIDDLE_Ablation.py --name MIDDLE-5 --randomSeed 2001
