#!/bin/bash
#SBATCH --job-name=cityscapes_analysis
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --output=cityscapes_analysis_%j.out

# Load required modules
module load 2021
module load Python/3.9.5-GCCcore-10.3.0

# Run the analysis script
python analyze_cityscapes.py