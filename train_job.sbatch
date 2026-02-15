#!/bin/bash
#SBATCH --job-name=aed_mae_vedant
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-80gb:1
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --output=training_log.txt
#SBATCH --error=training_error.txt
#SBATCH --cpus-per-task=4
echo "=== GPU Job Started ==="
echo "Job ID: $SLURM_JOB_ID"
echo "GPU allocated: $CUDA_VISIBLE_DEVICES"



# Loading the environment
source ~/.bashrc
module load cuda/12.8
# Activate environment with GPU support
conda activate pytorch_gpu
# Verify GPU access
echo "Python: $(which python)"

cd /nfs_home/users/poonam/vedant_project_folder/aed-mae/
# GPU-accelerated script
python main.py --dataset avenue
echo "=== GPU Job Completed ==="
