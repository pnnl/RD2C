#!/usr/bin/env bash
#SBATCH -A rd2c
#SBATCH -p dl
#SBATCH -t 1:30:00
#SBATCH -N 1
#SBATCH -n 10
#SBATCH -J middle_ablation-4

module purge
module load gcc/8.1.0
module load openmpi/4.1.4
source /people/born669/middle/bin/activate

mpirun -n 10 python MIDDLE_SingleAblation.py --name MIDDLE-4 --randomSeed 2382 --epochs 30 --L3 0.1
