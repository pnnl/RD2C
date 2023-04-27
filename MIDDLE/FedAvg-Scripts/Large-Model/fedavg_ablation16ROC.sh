#!/usr/bin/env bash
#SBATCH -A rd2c
#SBATCH -p dl
#SBATCH -t 3:30:00
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -J ma1
#SBATCH --job-name=fed-16roc-lm

module purge
module load gcc/8.1.0
module load openmpi/4.1.4
module load cuda/11.4
source /people/born669/middle/bin/activate

mpirun -n 16 python FedAvg.py --name lmFedAvg-1 --randomSeed 4282 --epochs 50 --graph_type clique-ring
mpirun -n 16 python FedAvg.py --name lmFedAvg-2 --randomSeed 42 --epochs 50 --graph_type clique-ring
mpirun -n 16 python FedAvg.py --name lmFedAvg-3 --randomSeed 1132 --epochs 50 --graph_type clique-ring
mpirun -n 16 python FedAvg.py --name lmFedAvg-4 --randomSeed 2382 --epochs 50 --graph_type clique-ring
mpirun -n 16 python FedAvg.py --name lmFedAvg-5 --randomSeed 91162 --epochs 50 --graph_type clique-ring