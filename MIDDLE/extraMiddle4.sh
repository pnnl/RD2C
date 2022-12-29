#!/usr/bin/env bash
#SBATCH -A rd2c
#SBATCH -p dl
#SBATCH -t 2:00:00
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -J ma1

module purge
module load gcc/8.1.0
module load openmpi/4.1.4
module load cuda/11.4
source /people/born669/middle/bin/activate

mpirun -n 8 python MIDDLE.py --name MIDDLE-1 --randomSeed 4282 --epochs 50 --L3 0
mpirun -n 8 python MIDDLE.py --name MIDDLE-2 --randomSeed 42 --epochs 50 --L3 0
mpirun -n 8 python MIDDLE.py --name MIDDLE-3 --randomSeed 1132 --epochs 50 --L3 0
mpirun -n 8 python MIDDLE.py --name MIDDLE-4 --randomSeed 2382 --epochs 50 --L3 0
mpirun -n 8 python MIDDLE.py --name MIDDLE-5 --randomSeed 9182 --epochs 50 --L3 0