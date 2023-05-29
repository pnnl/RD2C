#!/usr/bin/env bash
#SBATCH -A rd2c
#SBATCH -p dl
#SBATCH -t 30
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-node=1
#SBATCH -J test

module purge
module load python/3.7.0
module load cuda/11.1
source /people/born669/venv_fed/bin/activate

srun python tensorflow_test.py