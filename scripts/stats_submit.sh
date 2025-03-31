#!/bin/bash
#SBATCH --job-name=calculate_mhd_train_stats
#SBATCH --account=bdwx-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# #SBATCH --cpus-per-task=1
#SBATCH --gpus=1                              # Number of GPUs to allocate
#SBATCH --time=00:15:00
#SBATCH --mem=600G
#SBATCH --output=slurm_logs/mhd_stats_%j.log
#SBATCH --error=slurm_logs/mhd_stats_%j.err
# #SBATCH --nodelist=gpua057 # Optional, but recommended if there's availability
# #SBATCH --exclusive # Optional, but recommended if there's availability

module purge
module load cuda
module load python/miniforge3_pytorch/2.5.0.lua

source /sw/user/python/miniforge3-pytorch-2.5.0/etc/profile.d/conda.sh

conda activate NODs

python calculate_statistics.py --config ../configs/config_tfno3d.yaml

conda deactivate
