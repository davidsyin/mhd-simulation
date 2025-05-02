#!/bin/bash
#SBATCH --job-name=mhd_fno_train
#SBATCH --account=begd-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# #SBATCH --cpus-per-task=1
#SBATCH --gpus=1                              # Number of GPUs to allocate
#SBATCH --time=12:00:00
# #SBATCH --mem=600G
#SBATCH --output=slurm_logs/mhd_fno_train_%j.log
#SBATCH --error=slurm_logs/mhd_fno_train_%j.err
# #SBATCH --nodelist=gpua057 # Optional, but recommended if there's availability
# #SBATCH --exclusive # Optional, but recommended if there's availability

module purge
module load cuda
module load python/miniforge3_pytorch/2.5.0.lua

source /sw/user/python/miniforge3-pytorch-2.5.0/etc/profile.d/conda.sh

conda activate NODs
# wandb login

# Run test
# python ./run_training.py --config /u/skacmaz/NODs/configs/config_uno3d_Re100_deeper.yaml # --gpus 0,1,2
python ./run_training.py --config /u/skacmaz/NODs/configs/config_fno_Re100_deeper.yaml # --gpus 0,1,2
# python ./run_training.py --config /u/skacmaz/NODs/configs/config_fno_Re500.yaml # --gpus 0,1,2

conda deactivate
