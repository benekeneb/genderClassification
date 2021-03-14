#!/bin/bash
#SBATCH --mem=60G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpucloud
#SBATCH --gpus=2

module purge
module load spack_skylake_avx512
module load python/3.8-cuda-ml

#srun --mpi=pmi2 --pty --mem=40G -n1 --cpus-per-task=8 -p gpucloud --gpus=2 bash

#srun "cd /home/ga53kol/genderClassification"
#srun ls


srun python3 train.py
