#!/bin/bash
#SBATCH --job-name=NODs_diff_feat_gen
#SBATCH --account=bbjo-delta-gpu
#SBATCH --partition=gpuMI100x8-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# #SBATCH --cpus-per-task=1
# #SBATCH --gpus=1                              # Number of GPUs to allocate
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --mem=240G
#SBATCH --output=slurm_logs/NODs_diff_feat_gen_%j.log
#SBATCH --error=slurm_logs/NODs_diff_feat_gen_%j.err
# #SBATCH --nodelist=gpua057 # Optional, but recommended if there's availability
# #SBATCH --exclusive # Optional, but recommended if there's availability

module purge
module load anaconda3_mi100/23.7.4

source /sw/external/python/anaconda3_mi100_Sept2023/etc/profile.d/conda.sh

conda activate NODs_MI100
# wandb login

export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

python generate_diffusion_features.py --re 500


conda deactivate
