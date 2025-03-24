#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --partition=gpu_mig
#SBATCH --time=3:00:00

# Pull container from dockerhub
apptainer pull container.sif docker://tjmjaspers/nncv2025:v3