#!/bin/bash
#SBATCH --job-name=mhd_no_train
#SBATCH --account=bbjo-delta-gpu
#SBATCH --partition=gpuMI100x8-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# #SBATCH --cpus-per-task=1
#SBATCH --gpus=1                              # Number of GPUs to allocate
#SBATCH --time=02:00:00
#SBATCH --mem=240G
#SBATCH --output=slurm_logs/mhd_no_train_%j.log
#SBATCH --error=slurm_logs/mhd_no_train_%j.err
# #SBATCH --nodelist=gpua057 # Optional, but recommended if there's availability
# #SBATCH --exclusive # Optional, but recommended if there's availability

module purge
module load anaconda3_mi100/23.7.4

source /sw/external/python/anaconda3_mi100_Sept2023/etc/profile.d/conda.sh

conda activate NODs_MI100
# wandb login

# Run test
python ./run_training.py # --gpus 0,1,2

conda deactivate
