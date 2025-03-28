#!/bin/bash
#SBATCH --job-name=mhd_no_train
#SBATCH --account=bdwx-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                   # CPU cores per node -- max(num_workers)
#SBATCH --gpus=3                              # Number of GPUs to allocate
#SBATCH --time=00:10:00
#SBATCH --mem=600G
#SBATCH --output=slurm_logs/mhd_no_train_%j.log
#SBATCH --error=slurm_logs/mhd_no_train_%j.err
# #SBATCH --nodelist=gpua057 # Optional, but recommended if there's availability
# #SBATCH --exclusive # Optional, but recommended if there's availability

module purge
module load cuda
module load python/miniforge3_pytorch/2.5.0.lua

source /sw/user/python/miniforge3-pytorch-2.5.0/etc/profile.d/conda.sh

conda activate NODs
# wandb login

# Run test
python ./run_training.py --gpus 0,1,2

conda deactivate
