#!/usr/bin/env bash
#SBATCH -A rd2c
#SBATCH -p dl
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -J ma1
#SBATCH --job-name=mid-16ring-sm

module purge
module load gcc/8.1.0
module load openmpi/4.1.4
module load cuda/11.4
source /people/born669/middle/bin/activate

mpirun -n 16 python cifar10.py --name smMIDDLE-1 --randomSeed 4282 --epochs 100 --graph_type ring --large_model 0 --multi_model 0 --skew 0.5
mpirun -n 16 python cifar10.py --name smMIDDLE-2 --randomSeed 42 --epochs 100 --graph_type ring --large_model 0 --multi_model 0 --skew 0.5
mpirun -n 16 python cifar10.py --name smMIDDLE-3 --randomSeed 1132 --epochs 100 --graph_type ring --large_model 0 --multi_model 0 --skew 0.5
mpirun -n 16 python cifar10.py --name smMIDDLE-4 --randomSeed 2382 --epochs 100 --graph_type ring --large_model 0 --multi_model 0 --skew 0.5
mpirun -n 16 python cifar10.py --name smMIDDLE-5 --randomSeed 91162 --epochs 100 --graph_type ring --large_model 0 --multi_model 0 --skew 0.5