#!/bin/bash
#SBATCH --job-name=aed_mae_vedant_gradients
#SBATCH --partition=cpu-long
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH --output=training_log.txt
#SBATCH --error=training_error.txt
#SBATCH --cpus-per-task=5
echo "=== GPU Job Started ==="
echo "Job ID: $SLURM_JOB_ID"
echo "GPU allocated: $CUDA_VISIBLE_DEVICES"



# Loading the environment
source ~/.bashrc
# Activate environment with GPU support
conda activate pytorch_gpu

cd /nfs_home/users/poonam/vedant_project_folder/aed-mae/
python extract_gradients.py
echo "=== GPU Job Completed ==="