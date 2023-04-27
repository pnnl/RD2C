#!/usr/bin/env bash
#SBATCH -A rd2c
#SBATCH -p dl
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -J ma1
#SBATCH --job-name=fed-4r-sm

module purge
module load gcc/8.1.0
module load openmpi/4.1.4
module load cuda/11.4
source /people/born669/middle/bin/activate

mpirun -n 4 python cifar10_fedavg.py --name lmFedAvg-1 --randomSeed 4282 --epochs 200 --large_model 1 --skew 0.75
mpirun -n 4 python cifar10_fedavg.py --name lmFedAvg-2 --randomSeed 42 --epochs 200 --large_model 1 --skew 0.75
mpirun -n 4 python cifar10_fedavg.py --name lmFedAvg-3 --randomSeed 1132 --epochs 200 --large_model 1 --skew 0.75
mpirun -n 4 python cifar10_fedavg.py --name lmFedAvg-4 --randomSeed 2382 --epochs 200 --large_model 1 --skew 0.75
mpirun -n 4 python cifar10_fedavg.py --name lmFedAvg-5 --randomSeed 91162 --epochs 200 --large_model 1 --skew 0.75