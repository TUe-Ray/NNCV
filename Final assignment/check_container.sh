#!/bin/bash
#SBATCH --job-name=list-container-modules
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --partition=gpu_mig
#SBATCH --time=00:10:00
#SBATCH --output=list_modulesv7.out

# Load apptainer module if needed (only if system uses module system)
# module load apptainer


# Step 2: Check Python packages inside the container
echo "=== Python packages in container ==="
apptainer exec container.sif python3 -m pip list

# Optional: also check if conda is installed, and list packages
echo "=== Conda list (if available) ==="
apptainer exec container.sif bash -c "command -v conda && conda list || echo 'Conda not found'"

# Step 3: (Optional) Check environment modules (if module system exists inside container)
echo "=== Environment modules (if applicable) ==="
apptainer exec container.sif bash -c "command -v module && module avail || echo 'module command not found'"

# Done
echo "Done listing modules."
