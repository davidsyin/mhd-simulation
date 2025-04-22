#!/bin/bash
#SBATCH --job-name=run-module-tests
#SBATCH --account=bdwx-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# #SBATCH --cpus-per-task=1
#SBATCH --gpus=1                              # Number of GPUs to allocate
#SBATCH --time=00:02:00
# #SBATCH --mem=600G
#SBATCH --output=slurm_logs/tests_%j.log
#SBATCH --error=slurm_logs/tests_%j.err
# #SBATCH --nodelist=gpua057 # Optional, but recommended if there's availability
# #SBATCH --exclusive # Optional, but recommended if there's availability

module purge
module load cuda
module load python/miniforge3_pytorch/2.5.0.lua

source /sw/user/python/miniforge3-pytorch-2.5.0/etc/profile.d/conda.sh

conda activate NODs

python -m pytest tests/ -v

conda deactivate
