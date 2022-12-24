#!/usr/bin/env bash
#SBATCH -A rd2c
#SBATCH -p dl
#SBATCH -t 2:00:00
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -J ma3

module purge
module load gcc/8.1.0
module load openmpi/4.1.4
module load cuda/11.4
source /people/born669/middle/bin/activate

mpirun -n 16 python MIDDLE.py --name MIDDLE-3 --randomSeed 1132 --epochs 50 --L3 0.5
mpirun -n 16 python MIDDLE.py --name MIDDLE-3 --randomSeed 1132 --epochs 50 --L3 0.16666666666666666
mpirun -n 16 python MIDDLE.py --name MIDDLE-3 --randomSeed 1132 --epochs 50 --L3 0.6666666666666666
mpirun -n 16 python MIDDLE.py --name MIDDLE-3 --randomSeed 1132 --epochs 50 --L3 0.6
mpirun -n 16 python MIDDLE.py --name MIDDLE-3 --randomSeed 1132 --epochs 50
mpirun -n 16 python MIDDLE.py --name MIDDLE-3 --randomSeed 1132 --epochs 50 --L3 0.25
mpirun -n 16 python MIDDLE.py --name MIDDLE-3 --randomSeed 1132 --epochs 50 --L3 0.1
mpirun -n 16 python MIDDLE.py --name MIDDLE-3 --randomSeed 1132 --epochs 50 --L3 0.05
mpirun -n 16 python MIDDLE.py --name MIDDLE-3 --randomSeed 1132 --epochs 50 --L3 0